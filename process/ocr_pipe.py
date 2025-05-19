import os
import time
import subprocess
import struct
import cv2
import numpy as np
import pytesseract
import multiprocessing as mp
from fractions import Fraction
from tqdm import tqdm

VIDEO_PATH        = "./assets/two_games_check.mp4"   
HW_ACCEL          = "cuda"                   
INTERVAL_SECONDS  = 4                        
TARGET_HEIGHT     = 720     
ROI_HALF_HEIGHT   = 70                                      
NUM_OCR_WORKERS   = 4                        
QUEUE_MAXSIZE     = 32                      
KNOWN_MAPS = [
    "ABYSS","ASCENT","BIND","BREEZE","FRACTURE",
    "HAVEN","ICEBOX","PEARL","SPLIT","SUNSET"
]

def get_video_resolution(path):
    """
    Gets the video's resolution
    """
    cmd = ["ffprobe", "-v", "error",
           "-select_streams", "v:0",
           "-show_entries", "stream=width,height",
           "-of", "csv=p=0:s=x", path]
    out = subprocess.check_output(cmd, text=True).strip()  
    w, h = map(int, out.split("x"))
    return w, h

def clean_banner(bw):
    """
    bw  : binary uint8 (0/255) white-on-black
    out : cleaned binary, big glyphs preserved, thin lines removed
    """
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    thin_lines   = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    bw_no_rulers = cv2.subtract(bw, thin_lines)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw_no_rulers, 8)
    min_area = 100        # big blobs; tune once
    big = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            big[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    big = cv2.morphologyEx(big, cv2.MORPH_CLOSE, kernel, iterations=1)
    return big


def ocr_worker(frame_queue, result_queue):
    """
    Blocks the frame_queue and revieves (frame_idx, frame), where frame is 1D np(width, height, 3)
    Applies grayscale and threshold. Then runs OCR.
    Puts OCR Result into result_queue
    """
    while True:
        item = frame_queue.get()
        if item is None:
            break         
        frame_idx, frame_bgr = item
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, bw  = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        clean = clean_banner(bw)
        
        cv2.imwrite(f"./debug/debug_frame{frame_idx:03d}.png", clean)
        
        text = pytesseract.image_to_string(clean, config="--psm 7 --oem 3").upper()
        
        result_queue.put((frame_idx, text))

def stream_frames(video_path, interval_s, width, height, hw_accel, frame_queue, crop=None):
    """
    Spawns ffmpeg that sends raw BGR frames to stdout, one every interval_s seconds.
    Puts (frame_idx, numpy_frame) into frame_queue.
    """
    filter_chain = f"fps=1/{interval_s},scale=-2:{TARGET_HEIGHT}"
    ff_cmd = ["ffmpeg"]
    if hw_accel:
        ff_cmd += ["-hwaccel", hw_accel]
    ff_cmd += [
        "-i", video_path,
        "-vf", filter_chain,
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1"
    ]

    print("\n>> Spawning FFmpeg:", " ".join(ff_cmd))
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_per_frame = width * height * 3   # 24‑bit BGR
    frame_idx = 0
    pbar = tqdm(desc="Reading frames", unit="frame")

    try:
        while True:
            raw = proc.stdout.read(bytes_per_frame)
            if len(raw) < bytes_per_frame:
                break  
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            if crop is not None:
                top, bot, x1, x2 = crop
                frame = frame[top:bot, x1:x2]        

            frame_queue.put((frame_idx, frame))
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        proc.stdout.close()
        proc.wait()
        for _ in range(NUM_OCR_WORKERS):
            frame_queue.put(None)

def parse_results(result_queue, total_frames):
    last_map = None
    games = []
    received = 0

    pbar = tqdm(total=total_frames, desc="Parsing OCR")
    while received < total_frames:
        frame_idx, text = result_queue.get()
        received += 1
        pbar.update(1)

        for m in KNOWN_MAPS:
            if m in text:
                last_map = (m, frame_idx)
                break

        if "VICTORY" in text or "DEFEAT" in text:
            if last_map:
                result = "VICTORY" if "VICTORY" in text else "DEFEAT"
                games.append((last_map[0], last_map[1], frame_idx, result))
                last_map = None
    pbar.close()
    return games

def main():
    vid_w, vid_h = get_video_resolution(VIDEO_PATH)

    y_top    = max(0, vid_h//2 - ROI_HALF_HEIGHT)
    y_bottom = min(vid_h, vid_h//2 + ROI_HALF_HEIGHT)
    x1 = int(0.25 * vid_w)
    x2 = int(0.75 * vid_w)

    
    frame_queue  = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = mp.Queue()

    # start OCR workers
    workers = [mp.Process(target=ocr_worker, args=(frame_queue, result_queue))
               for _ in range(NUM_OCR_WORKERS)]
    for w in workers: w.start()

    t0 = time.time()
    # read / stream frames (blocking until done)
    stream_frames(VIDEO_PATH, INTERVAL_SECONDS,
                  vid_w, vid_h, HW_ACCEL, frame_queue,
                  crop=(y_top, y_bottom, x1, x2))

    
    t1 = time.time()

    total_frames = result_queue.qsize()  
    games = parse_results(result_queue, total_frames)
    t2 = time.time()

    for w in workers: w.join()

    print("\n=== Completed ===")
    print(f"Frames decoded / pushed : {total_frames}")
    print(f"FFmpeg + decode time    : {t1 - t0:.2f} s")
    print(f"OCR  + parsing time     : {t2 - t1:.2f} s")
    print("\nDetected games:")
    for m, start_f, end_f, res in games:
        print(f"  {m}: {res}  (start frame {start_f}, end frame {end_f})")

if __name__ == "__main__":
    main()

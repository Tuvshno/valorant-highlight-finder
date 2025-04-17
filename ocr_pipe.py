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

VIDEO_PATH        = "two_games_check.mp4"   
HW_ACCEL          = "cuda"                   
INTERVAL_SECONDS  = 4                        
TARGET_HEIGHT     = 720     
ROI_HALF_HEIGHT   = 150                                      
NUM_OCR_WORKERS   = 4                        
QUEUE_MAXSIZE     = 32                       # 32 * (720x1280*3bytes) = 86MB of RAM -> Crop = 32MB
KNOWN_MAPS = [
    "ABYSS","ASCENT","BIND","BREEZE","FRACTURE",
    "HAVEN","ICEBOX","PEARL","SPLIT","SUNSET"
]

def get_video_resolution(path):
    """
    Gets the video's resolution then calculatesm new height and width based on TARGET_HEIGHT.
    Returns the targeted resolution height, width
    """
    cmd = ["ffprobe", "-v", "error",
           "-select_streams", "v:0",
           "-show_entries", "stream=width,height",
           "-of", "csv=p=0:s=x", path]
    out = subprocess.check_output(cmd, text=True).strip()  # e.g. "1920x1080"
    w, h = map(int, out.split("x"))
    new_h = TARGET_HEIGHT
    new_w = int(round(w * TARGET_HEIGHT / h / 2) * 2)
    return new_w, new_h

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
        _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(f"debug_frame{frame_idx:03d}.png", thr)
        
        text = pytesseract.image_to_string(thr, config="--psm 7 --oem 3").upper()
        
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
                top, bottom = crop
                frame = frame[top:bottom, :]
                
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
    width, height = get_video_resolution(VIDEO_PATH)
    
    roi_top    = max(0, height//2 - ROI_HALF_HEIGHT)
    roi_bottom = min(height, height//2 + ROI_HALF_HEIGHT)
    roi_height = roi_bottom - roi_top
    
    frame_queue  = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = mp.Queue()

    # start OCR workers
    workers = [mp.Process(target=ocr_worker, args=(frame_queue, result_queue))
               for _ in range(NUM_OCR_WORKERS)]
    for w in workers: w.start()

    t0 = time.time()
    # read / stream frames (blocking until done)
    stream_frames(VIDEO_PATH, INTERVAL_SECONDS, width, height, HW_ACCEL, frame_queue, crop=(roi_top, roi_bottom))
    
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

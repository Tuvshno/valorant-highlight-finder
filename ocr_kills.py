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
QUEUE_MAXSIZE     = 32                       # 32 * (720x1280*3bytes) = 86MB of RAM -> Crop = 32MB

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
    target_bgr = np.array([98, 215, 206], dtype=np.uint8)
    TOLERANCE   = 10             
    lower = cv2.subtract(target_bgr, np.array([TOLERANCE]*3, dtype=np.uint8))
    upper = cv2.add   (target_bgr, np.array([TOLERANCE]*3, dtype=np.uint8))
    while True:
        item = frame_queue.get()
        if item is None:
            break         
        frame_idx, frame_bgr = item
        mask = cv2.inRange(frame_bgr, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            x1 = max(x - 20, 0)
            y1 = max(y, 0)
            x2 = min(x + w + 20, frame_bgr.shape[1])
            y2 = min(y + h + 10, frame_bgr.shape[0])

            crop = frame_bgr[y1:y2, x1:x2]
            
            # _, bw  = cv2.threshold(crop, 220, 255, cv2.THRESH_BINARY)
            
            hex_colors = ['#B5F0DE', '#AFEFDD', '#A8E3D1', '#ACE2A7']

            target_bgrs = []
            for hx in hex_colors:
                # strip ‘#’, parse as R,G,B, then reverse for BGR
                r, g, b = tuple(int(hx[i:i+2], 16) for i in (1, 3, 5))
                target_bgrs.append(np.array([b, g, r], dtype=np.uint8))

            TOL = 10

            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            for bgr in target_bgrs:
                lower = cv2.subtract(bgr, np.array([TOL]*3, dtype=np.uint8))
                upper = cv2.add   (bgr, np.array([TOL]*3, dtype=np.uint8))
                m = cv2.inRange(crop, lower, upper)
                mask = cv2.bitwise_or(mask, m)

            # cv2.imwrite(f"./debug/debug_frame{frame_idx:03d}.png", frame_bgr)
            cv2.imwrite(f"./debug/debug_frame{frame_idx:03d}.png", mask)
        
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
    received = 0

    pbar = tqdm(total=total_frames, desc="Parsing OCR")
    while received < total_frames:
        frame_idx, text = result_queue.get()
        received += 1
        pbar.update(1)
        print(text)

    pbar.close()

def main():
    vid_w, vid_h = get_video_resolution(VIDEO_PATH)

    y_top    = 0
    y_bottom = min(vid_h, vid_h//3)
    x1 = int(0.75 * vid_w)
    x2 = int(vid_w)

    
    frame_queue  = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = mp.Queue()

    workers = [mp.Process(target=ocr_worker, args=(frame_queue, result_queue))
               for _ in range(NUM_OCR_WORKERS)]
    for w in workers: w.start()

    t0 = time.time()
    stream_frames(VIDEO_PATH, INTERVAL_SECONDS,
                  vid_w, vid_h, HW_ACCEL, frame_queue,
                  crop=(y_top, y_bottom, x1, x2))

    
    t1 = time.time()

    total_frames = result_queue.qsize()  
    parse_results(result_queue, total_frames)
    t2 = time.time()

    for w in workers: w.join()

    print("\n=== Completed ===")
    print(f"Frames decoded / pushed : {total_frames}")
    print(f"FFmpeg + decode time    : {t1 - t0:.2f} s")
    print(f"OCR  + parsing time     : {t2 - t1:.2f} s")
    print("\nDetected games:")

if __name__ == "__main__":
    main()

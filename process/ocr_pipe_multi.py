import os
import time
import subprocess
import cv2
import numpy as np
import pytesseract
import multiprocessing as mp
from fractions import Fraction
from tqdm import tqdm

VIDEO_PATH       = "two_games_check.mp4"
HW_ACCEL         = "cuda"        
INTERVAL_SECONDS = 4             
TARGET_HEIGHT    = 720
ROI_HALF_HEIGHT  = 150           
NUM_OCR_WORKERS  = 4
QUEUE_MAXSIZE    = 32           

KNOWN_MAPS = [
    "ABYSS","ASCENT","BIND","BREEZE","FRACTURE",
    "HAVEN","ICEBOX","PEARL","SPLIT","SUNSET"
]

def get_video_resolution(path):
    """Probe video, compute scaled width,height for TARGET_HEIGHT."""
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","v:0",
        "-show_entries","stream=width,height",
        "-of","csv=p=0:s=x", path
    ]
    out = subprocess.check_output(cmd, text=True).strip() 
    orig_w, orig_h = map(int, out.split("x"))
    new_h = TARGET_HEIGHT
    new_w = int(round(orig_w * new_h / orig_h / 2) * 2)
    return new_w, new_h

def ocr_worker(frame_queue, result_queue):
    """
    Each item is (frame_idx, crops_dict).
    crops_dict has keys: 'map','timer','score_L','score_R' with BGR images.
    We run region-specific OCR and emit a dict.
    """
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame_idx, crops = item

        out = {"frame": frame_idx}

        # --- MAP (uppercase letters) ---
        gray = cv2.cvtColor(crops["map"], cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(
            thr,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip().upper()
        out["map"] = txt

        gray = cv2.cvtColor(crops["timer"], cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(
            thr,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:"
        ).strip()
        out["timer"] = txt

        gray = cv2.cvtColor(crops["score_L"], cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(
            thr,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
        ).strip()
        out["score_L"] = txt

        gray = cv2.cvtColor(crops["score_R"], cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(
            thr,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
        ).strip()
        out["score_R"] = txt

        result_queue.put(out)

def stream_frames(video_path, interval_s, width, height, hw_accel, rois, frame_queue):
    """
    Decode+scale+sample via FFmpeg → raw BGR → pipe → crop each ROI → queue.
    """
    filter_chain = f"fps=1/{interval_s},scale=-2:{height}"
    cmd = ["ffmpeg"]
    if hw_accel:
        cmd += ["-hwaccel", hw_accel]
    cmd += [
        "-i", video_path,
        "-vf", filter_chain,
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    bytes_per_frame = width * height * 3
    idx = 0
    pbar = tqdm(desc="Reading frames", unit="frame")
    try:
        while True:
            raw = proc.stdout.read(bytes_per_frame)
            if len(raw) < bytes_per_frame:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            crops = {}
            for name,(y1,y2,x1,x2) in rois.items():
                crops[name] = frame[y1:y2, x1:x2]

            frame_queue.put((idx, crops))
            idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        proc.stdout.close()
        proc.wait()
        for _ in range(NUM_OCR_WORKERS):
            frame_queue.put(None)

def parse_results(result_queue, total_frames):
    """
    Collect all OCR records, then detect games by looking for map + Victory/Defeat.
    """
    last_map = None
    games = []
    received = 0
    pbar = tqdm(total=total_frames, desc="Parsing OCR")
    while received < total_frames:
        rec = result_queue.get()
        received += 1
        pbar.update(1)

        m = rec["map"]
        if m in KNOWN_MAPS:
            last_map = (m, rec["frame"])

        text = m  
        if "VICTORY" in text or "DEFEAT" in text:
            if last_map:
                result = "VICTORY" if "VICTORY" in text else "DEFEAT"
                games.append((last_map[0], last_map[1], rec["frame"], result))
                last_map = None
    pbar.close()
    return games

def main():
    width, height = get_video_resolution(VIDEO_PATH)

    y_center = height // 2
    ROIS = {
        "map":     (y_center-ROI_HALF_HEIGHT, y_center+ROI_HALF_HEIGHT, 0, width),
        "timer":   (int(0.03*height), int(0.11*height),
                    width//2 - int(0.05*width)//2,
                    width//2 + int(0.05*width)//2),
        "score_L": (int(0.03*height), int(0.11*height),
                    # left of timer by 3% gap
                    width//2 - int(0.05*width)//2 - int(0.03*width) - int(0.03*width),
                    width//2 - int(0.05*width)//2 - int(0.03*width)),
        "score_R": (int(0.03*height), int(0.11*height),
                    width//2 + int(0.05*width)//2 + int(0.03*width),
                    width//2 + int(0.05*width)//2 + int(0.03*width) + int(0.03*width))
    }

    print("ROIs (y1,y2,x1,x2):")
    for k,v in ROIS.items():
        print(f"  {k}: {v}")

    # Set up queues & workers
    frame_q  = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_q = mp.Queue()

    workers = [ mp.Process(target=ocr_worker, args=(frame_q, result_q))
                for _ in range(NUM_OCR_WORKERS) ]
    for w in workers:
        w.start()

    t0 = time.time()
    stream_frames(VIDEO_PATH, INTERVAL_SECONDS,
                  width, height, HW_ACCEL, ROIS, frame_q)
    t1 = time.time()

    total = result_q.qsize()
    games = parse_results(result_q, total)
    t2 = time.time()

    for w in workers:
        w.join()

    print("\n=== Completed ===")
    print(f"Frames decoded+cropped : {total}")
    print(f"FFmpeg+decode time     : {t1-t0:.2f}s")
    print(f"OCR+parse time         : {t2-t1:.2f}s\n")
    print("Detected games:")
    for m,sf,ef,res in games:
        print(f"  Map={m}, Result={res}, start_frame={sf}, end_frame={ef}")

if __name__ == "__main__":
    main()

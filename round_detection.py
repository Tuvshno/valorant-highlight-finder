import os, time, subprocess, cv2, numpy as np
import pytesseract, multiprocessing as mp
from tqdm import tqdm

VIDEO_PATH       = "./assets/two_games_check.mp4"
HW_ACCEL         = "cuda"
INTERVAL_SECONDS = 360
TARGET_HEIGHT    = 720
ROI_HALF_HEIGHT  = 150
NUM_OCR_WORKERS  = 4
QUEUE_MAXSIZE    = 32

KNOWN_MAPS = [
    "ABYSS","ASCENT","BIND","BREEZE","FRACTURE",
    "HAVEN","ICEBOX","PEARL","SPLIT","SUNSET"
]


def get_video_resolution(path):
    out = subprocess.check_output(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height","-of","csv=p=0:s=x",path],
        text=True).strip()
    w,h = map(int, out.split("x"))
    new_h = TARGET_HEIGHT
    new_w = int(round(w*new_h/h/2)*2)
    return new_w, new_h


def ocr_roi(img_bgr, wl):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cfg = f"--psm 7 --oem 3 -c tessedit_char_whitelist={wl}"
    return pytesseract.image_to_string(thr, config=cfg).strip()

def ocr_worker(frame_q, result_q):
    while True:
        item = frame_q.get()
        if item is None:
            break
        idx, crops = item                      

        rec = {"frame": idx}
        rec["map"]     = ocr_roi(crops["map"],     "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        rec["timer"]   = ocr_roi(crops["timer"],   "0123456789:")
        rec["score_L"] = ocr_roi(crops["score_L"], "0123456789/")
        rec["score_R"] = ocr_roi(crops["score_R"], "0123456789/")
        result_q.put(rec)


def stream_frames(path, interval_s, w, h, hw_accel, rois, frame_q):
    filter_chain = f"fps=1/{interval_s},scale=-2:{h}"
    cmd = ["ffmpeg"]
    if hw_accel:
        cmd += ["-hwaccel", hw_accel]
    cmd += ["-i", path, "-vf", filter_chain,
            "-pix_fmt","bgr24","-f","rawvideo","pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_per = w*h*3
    idx = 0
    for _ in tqdm(iter(int,1), desc="Frames"):      # infinite loop broken inside
        raw = proc.stdout.read(bytes_per)
        if len(raw) < bytes_per:
            break
        frame = np.frombuffer(raw, np.uint8).reshape((h,w,3))

        crops = { name: frame[y1:y2, x1:x2] for name,(y1,y2,x1,x2) in rois.items() }
        frame_q.put((idx, crops))
        idx += 1
    proc.stdout.close(); proc.wait()
    for _ in range(NUM_OCR_WORKERS):
        frame_q.put(None)


def parse_results(res_q, total):
    last_map = None
    games = []
    got = 0
    for _ in tqdm(range(total), desc="Parsing"):
        rec = res_q.get()
        got += 1

        if rec["map"] in KNOWN_MAPS:
            last_map = (rec["map"], rec["frame"])

        banner_txt = rec["map"]   # map ROI may catch VICTORY text
        if "VICTORY" in banner_txt or "DEFEAT" in banner_txt:
            if last_map:
                result = "VICTORY" if "VICTORY" in banner_txt else "DEFEAT"
                games.append((last_map[0], last_map[1], rec["frame"], result))
                last_map = None

        # demo‑print timer/score (remove if noisy)
        print(f"f{rec['frame']:05d}  timer={rec['timer']}  "
              f"{rec['score_L']}-{rec['score_R']}")
    return games


def main():
    vid_w, vid_h = get_video_resolution(VIDEO_PATH)

    y1 = int(0.03*vid_h); y2 = int(0.11*vid_h)
    timer_w = int(0.05*vid_w)
    cx = vid_w//2; tx1,tx2 = cx-timer_w//2, cx+timer_w//2
    round_w = int(0.03*vid_w); gap = int(0.03*vid_w)
    lrx1,lrx2 = tx1-gap-round_w, tx1-gap
    rrx1,rrx2 = tx2+gap, tx2+gap+round_w
    
    ROIS = {
        "map":     (vid_h//2-ROI_HALF_HEIGHT, vid_h//2+ROI_HALF_HEIGHT, 0, vid_w),
        "timer":   (y1,y2,tx1,tx2),
        "score_L": (y1,y2,lrx1,lrx2),
        "score_R": (y1,y2,rrx1,rrx2)
    }

    frame_q  = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_q = mp.Queue()

    worker_procs = [
        mp.Process(target=ocr_worker, args=(frame_q, result_q))
        for _ in range(NUM_OCR_WORKERS)
    ]
    for p in worker_procs:
        p.start()

    t0 = time.time()
    stream_frames(VIDEO_PATH, INTERVAL_SECONDS,
                  vid_w, vid_h, HW_ACCEL, ROIS, frame_q)
    t1 = time.time()

    total = result_q.qsize()
    games = parse_results(result_q, total)
    t2 = time.time()

    for p in worker_procs:
        p.join()

    print("\n=== Completed ===")
    print(f"Frames processed   : {total}")
    print(f"Decode/queue time  : {t1-t0:.2f} s")
    print(f"OCR + parse time   : {t2-t1:.2f} s\n")
    print("Detected games:")
    for m,sf,ef,res in games:
        print(f"  {m}: {res}  start={sf}  end={ef}")

if __name__ == "__main__":
    main()

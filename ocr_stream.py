import cv2
import pytesseract
import multiprocessing
from tqdm import tqdm
import time

def ocr_worker(task_queue, result_queue):
    """A worker that reads (frame_index, BGR_image) from task_queue, does OCR, and puts results."""
    while True:
        item = task_queue.get()
        if item is None:
            break
        
        frame_index, roi_bgr = item
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        recognized_text = pytesseract.image_to_string(thresh, config="--psm 7 --oem 3").upper()
        
        result_queue.put((frame_index, recognized_text))

def main():
    video_path = "defeat.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = max(1, int(fps*4)) if fps else 1
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_center = height // 2
    top = max(0, y_center - 150)
    bottom = min(height, y_center + 150)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start = time.time()
    
    # Create queues
    task_queue = multiprocessing.Queue(maxsize=50)  
    result_queue = multiprocessing.Queue()
        
    # Start worker processes
    num_workers = 4
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=ocr_worker, args=(task_queue, result_queue))
        p.start()
        processes.append(p)
    
    frame_index = 0
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % skip_rate == 0:
            roi = frame[top:bottom, 0:width]
            task_queue.put((frame_index, roi))
        frame_index += 1

    cap.release()
    
    for _ in processes:
        task_queue.put(None)
    num_tasks = frame_index // skip_rate + 1  # or a more accurate count
    results_collected = 0
    
    results = []
    
    while results_collected < num_tasks:
        try:
            fidx, text = result_queue.get(timeout=3)
            results.append((fidx, text))
            results_collected += 1
        except:
            pass
    
    for p in processes:
        p.join()
    
    results.sort(key=lambda x: x[0])
    
    last_map = None
    last_map_frame = None
    known_maps = ["ABYSS", "ASCENT", "BIND", "BREEZE", "FRACTURE",
                  "HAVEN", "ICEBOX", "PEARL", "SPLIT", "SUNSET"]
    games_data = []
    
    for (fidx, recognized_text) in results:
        map_detected = None
        for m in known_maps:
            if m in recognized_text:
                map_detected = m
                last_map = m
                last_map_frame = fidx
                print(f"[Frame {fidx}] Map = {m}")
                break
        
        if "VICTORY" in recognized_text or "DEFEAT" in recognized_text:
            if last_map is not None:
                result = "VICTORY" if "VICTORY" in recognized_text else "DEFEAT"
                games_data.append({
                    "map": last_map,
                    "start_frame": last_map_frame,
                    "end_frame": fidx,
                    "result": result
                })
                print(f"[Frame {fidx}] {result} for map {last_map}")
                last_map = None
                last_map_frame = None
    
    print("=== Detected Games ===")
    for i, g in enumerate(games_data, start=1):
        print(f"Game {i}: {g}")

    end = time.time()
    print(end - start)
    
if __name__ == "__main__":
    main()

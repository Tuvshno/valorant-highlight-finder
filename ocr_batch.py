import cv2
import pytesseract
import multiprocessing
from tqdm import tqdm
import time

def process_frame_for_ocr(args):
    """
    Worker function that receives (frame_index, cropped_image) and returns
    recognized text plus the frame index.
    """
    frame_index, cropped_bgr = args
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    recognized_text = pytesseract.image_to_string(thresh, config="--psm 7 --oem 3").upper()
    return (frame_index, recognized_text)

def main():
    video_path = 'vod 2.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = max(1, int(fps * 4)) if fps else 1  # 1 frame every 4s if fps=30
    
    # Figure out cropping region (±150 px from center)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_center = height // 2
    top = max(0, y_center - 150)
    bottom = min(height, y_center + 150)
    
    # We'll collect (frame_index, cropped_roi) pairs here
    frames_to_ocr = []
    
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = time.time()
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % skip_rate == 0:
            # Crop ROI
            roi = frame[top:bottom, 0:width]
            # You can do the 'difference check' here if you want, to skip repeated frames
            # We'll keep it simple for now
            frames_to_ocr.append((frame_index, roi))
        
        frame_index += 1

    cap.release()
    
    # Now we do OCR in parallel
    print(f"Collected {len(frames_to_ocr)} frames to process.")
    
    # Create a multiprocessing pool
    num_workers = 4  # adjust based on your CPU
    pool = multiprocessing.Pool(num_workers)
    
    # Map all frames to OCR in parallel
    results = list(tqdm(pool.imap(process_frame_for_ocr, frames_to_ocr), total=len(frames_to_ocr), desc="OCR"))
    
    pool.close()
    pool.join()
    
    # 'results' is a list of (frame_index, recognized_text) in the order they were processed
    # We can sort by frame_index if needed
    results.sort(key=lambda x: x[0])
    
    # Now parse recognized_text for map detection, victory/defeat, etc.
    # The parsing logic can be the same as before, but done in a single pass over 'results'.
    last_map = None
    last_map_frame = None
    games_data = []
    
    known_maps = ["ABYSS", "ASCENT", "BIND", "BREEZE", "FRACTURE",
                  "HAVEN", "ICEBOX", "PEARL", "SPLIT", "SUNSET"]
    
    for (fidx, text) in results:
        # Map check
        for m in known_maps:
            if m in text:
                last_map = m
                last_map_frame = fidx
                print(f"[Frame {fidx}] Detected map: {m}")
                break
        
        # Victory/Defeat check
        if "VICTORY" in text or "DEFEAT" in text:
            if last_map is not None:
                result = "VICTORY" if "VICTORY" in text else "DEFEAT"
                # record game
                game_entry = {
                    "map": last_map,
                    "start_frame": last_map_frame,
                    "end_frame": fidx,
                    "result": result
                }
                games_data.append(game_entry)
                print(f"[Frame {fidx}] {result} detected for map {last_map}!")
                # reset
                last_map = None
                last_map_frame = None
    
    print("\n=== Detected Games ===")
    for i, g in enumerate(games_data, start=1):
        print(f"Game {i}: {g}")

    end = time.time()
    print(end - start)
    
if __name__ == "__main__":
    main()

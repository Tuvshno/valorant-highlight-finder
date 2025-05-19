import os
import subprocess
import glob
import cv2
import pytesseract
import multiprocessing
import time
from fractions import Fraction
from tqdm import tqdm

VIDEO_PATH = "performance_test.mp4"
HW_ACCEL = "cuda"
OUTPUT_FOLDER = "frames_out2"
NUM_WORKERS = 4
INTERVAL_SECONDS = 4

def extract_fps_ffmpeg(video_path):
    """
    Use ffprobe to get the r_frame_rate (e.g. '30000/1001'),
    convert it to a float, and round to int.
    """
    
    # frame-based extraction and no resolution change -> 20 minutes for 3 hours
    ffmpeg_command = [
        "ffprobe",
        "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]

    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    fraction_str = result.stdout.strip()  # e.g. "30000/1001"
    fps_float = float(Fraction(fraction_str))  # e.g. ~29.97
    fps_int = round(fps_float)
    return fps_int

def extract_frames_ffmpeg(video_path, out_folder, skip, hw_accel=None):
    """
    Calls FFmpeg to extract 1 frame every 'skip' frames 
    using hardware acceleration if specified.
    
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    # Build FFmpeg command
    ffmpeg_command = ["ffmpeg"]
    if hw_accel:
        ffmpeg_command += ["-hwaccel", hw_accel]

    ffmpeg_command += [
        "-i", video_path,
        "-vf", f"select='not(mod(n,{skip}))',setpts=N/FRAME_RATE/TB",
        "-vsync", "0",
        os.path.join(out_folder, "frame_%06d.png")
    ]

    print("Running FFmpeg command:\n", " ".join(ffmpeg_command))
    subprocess.run(ffmpeg_command, check=True)
    print("FFmpeg extraction complete.")

def extract_frames_time_based(video_path, out_folder, interval_s=4, hw_accel=None):
    """
    Extract 1 frame every `interval_s` seconds using time-based FPS filter,
    scale to 720p height, and output to JPEG.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
        
    filter_str = f"fps=1/{interval_s},scale=-2:720"

    ffmpeg_cmd = ["ffmpeg"]
    if hw_accel:
        ffmpeg_cmd += ["-hwaccel", hw_accel]  
    
    ffmpeg_cmd += [
        "-i", video_path,
        "-vf", filter_str,
        "-vsync", "0",
        os.path.join(out_folder, "frame_%06d.jpg")
    ]
    
    print("Running FFmpeg command:\n", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)
    print("Time-based extraction complete.")



def ocr_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return (image_path, "")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, config="--psm 7 --oem 3").upper()
    return (image_path, text)

KNOWN_MAPS = [
    "ABYSS", "ASCENT", "BIND", "BREEZE", "FRACTURE",
    "HAVEN", "ICEBOX", "PEARL", "SPLIT", "SUNSET"
]

def parse_detections(ocr_results):
    last_map = None
    last_map_path = None
    games_data = []

    for (img_path, text) in ocr_results:
        # Check if there's a known map
        for m in KNOWN_MAPS:
            if m in text:
                last_map = m
                last_map_path = img_path
                print(f"[MAP] {m} => {img_path}")
                break

        # Check for victory/defeat
        if "VICTORY" in text or "DEFEAT" in text:
            if last_map:
                result = "VICTORY" if "VICTORY" in text else "DEFEAT"
                games_data.append({
                    "map": last_map,
                    "start_frame_image": last_map_path,
                    "end_frame_image": img_path,
                    "result": result
                })
                print(f"==> {result} on map {last_map}, from {last_map_path} to {img_path}")
                last_map = None
                last_map_path = None
            else:
                print(f"Victory/Defeat at {img_path} but no map known yet.")
    return games_data

def run_frame_based_approach():
    # Determine FPS and define skip
    fps = extract_fps_ffmpeg(VIDEO_PATH)
    skip_frames = int(fps * 4)   # for 1 frame every ~4 seconds
    print(f"Video FPS={fps}, skip={skip_frames}")

    # Extract frames with FFmpeg
    extract_frames_ffmpeg(VIDEO_PATH, OUTPUT_FOLDER, skip_frames, hw_accel=HW_ACCEL)
    frame_files =  sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.png")))
    if not frame_files:
        print(f"No frames found in {OUTPUT_FOLDER}. Check FFmpeg step.")
        return
    return frame_files
    
def run_time_based_approach():
    extract_frames_time_based(
        video_path=VIDEO_PATH,
        out_folder=OUTPUT_FOLDER,
        interval_s=INTERVAL_SECONDS,
        hw_accel=HW_ACCEL
    )

    # 2) Gather extracted .jpg frames
    frame_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.jpg")))
    if not frame_files:
        print(f"No frames found in {OUTPUT_FOLDER}. Check FFmpeg step.")
        return
    return frame_files

def performance_test():
    VIDEO_PATH = "performance_test.mp4"
    OUTPUT_FOLDER = "performance_spam"
    
    print("Comparing frame-based vs time-based extraction on:", VIDEO_PATH)

    # 1) Frame-based
    start_fb = time.time()
    fb_frames = run_frame_based_approach()
    run_ocr(fb_frames)
    end_fb = time.time()
    print(f"Frame-based extracted {len(fb_frames)} frames in {end_fb - start_fb:.2f} seconds.\n")

    # 2) Time-based
    start_tb = time.time()
    tb_frames = run_time_based_approach()
    run_ocr(tb_frames)
    end_tb = time.time()
    print(f"Time-based extracted {len(tb_frames)} frames in {end_tb - start_tb:.2f} seconds.\n")

def run_ocr(frame_files):
    if not frame_files:
        print(f"No frames found in {OUTPUT_FOLDER}. Check FFmpeg step.")
        return

    print(f"Found {len(frame_files)} frame(s) to OCR...")

    # Parallel OCR
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(ocr_image, frame_files),
                            total=len(frame_files),
                            desc="OCR"))

    # Sort results if needed
    results.sort(key=lambda x: x[0])

    # Parse for map & victory/defeat
    games_data = parse_detections(results)

    print("\n=== Final Detected Games ===")
    for i, g in enumerate(games_data, start=1):
        print(f"Game {i}: Map={g['map']}, Start={g['start_frame_image']}, End={g['end_frame_image']}, Result={g['result']}")

    

def main():
    # performance_test()
    # return
    
    start = time.time()

    # frame_files = run_frame_based_approach()
    frame_files = run_time_based_approach()
    
    

if __name__ == "__main__":
    main()

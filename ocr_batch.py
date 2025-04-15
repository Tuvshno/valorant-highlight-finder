import os
import subprocess
import glob
import cv2
import pytesseract
import multiprocessing
import time
from fractions import Fraction
from tqdm import tqdm

VIDEO_PATH = "vod 2.mp4"
HW_ACCEL = "cuda"
OUTPUT_FOLDER = "frames_out"
NUM_WORKERS = 4

def extract_fps_ffmpeg(video_path):
    """
    Use ffprobe to get the r_frame_rate (e.g. '30000/1001'),
    convert it to a float, and round to int.
    """
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


def main():
    start = time.time()

    # Determine FPS and define skip
    fps = extract_fps_ffmpeg(VIDEO_PATH)
    skip_frames = int(fps * 4)   # for 1 frame every ~4 seconds
    print(f"Video FPS={fps}, skip={skip_frames}")

    # Extract frames with FFmpeg
    extract_frames_ffmpeg(VIDEO_PATH, OUTPUT_FOLDER, skip_frames, hw_accel=HW_ACCEL)

    # Gather frames
    frame_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.png")))
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

    end = time.time()
    print("Elapsed time:", end - start)

if __name__ == "__main__":
    main()

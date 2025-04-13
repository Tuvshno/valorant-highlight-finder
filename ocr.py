import cv2
import pytesseract
from tqdm import tqdm

# If needed, specify the Tesseract install path on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_max_frames(video_path):
    """Return total number of frames in the video or -1 if not accessible."""
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        return -1
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()
    return total

video_path = 'clip.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get FPS (frames per second) to compute skip_rate
fps = cap.get(cv2.CAP_PROP_FPS)
skip_rate = max(1, int(fps / 10)) if fps else 1  # E.g., process ~10 frames/second

total_frames = get_max_frames(video_path)
if total_frames == -1:
    print("Error: Could not determine total frames.")
    cap.release()
    exit()

# Get video dimensions
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Define the vertical center band: 300px above & below center
y_center = height // 2
top     = max(0, y_center - 150)
bottom  = min(height, y_center + 150)

# Tesseract config
config = "--psm 7 --oem 3"  # single line, LSTM engine
threshold_value = 150

frame_index = 0

# Iterate through frames, showing a progress bar with tqdm
for _ in tqdm(range(total_frames), desc="Processing"):
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every skip_rate-th frame for speed
    if frame_index % skip_rate == 0:
        # Crop the frame: entire width (0:width), but only from top to bottom
        roi = frame[top:bottom, 0:width]

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # OCR the thresholded region
        recognized_text = pytesseract.image_to_string(thresh, config=config)

        # Check if "FRACTURE" appears in the recognized text
        if "FRACTURE" in recognized_text.upper():
            print(f"[Frame {frame_index}] FRACTURE detected!")
            # Show the cropped frame for confirmation
            cv2.imshow("Cropped ROI (FRACTURE)", roi)
        else:
            # If you just want to see the cropped region each iteration, uncomment:
            cv2.imshow("Cropped ROI", roi)

        # Small delay so the imshow window updates
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

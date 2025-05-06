import cv2
import torch
from ultralytics import YOLO

VIDEO_PATH = "./assets/highlight.webm"
DEBUG_DIR  = "./debug"
SKIP_SEC   = 4
BATCH_SIZE = 8    

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load model
model = YOLO("./best.pt")
model.to(device)
if device.startswith("cuda"):
    model.fuse()
    model.model.half()

cap = cv2.VideoCapture(VIDEO_PATH)
fps       = cap.get(cv2.CAP_PROP_FPS)
skip      = int(fps * SKIP_SEC)

frames, indices = [], []
frame_idx = 0

while True:
    ret = cap.grab()
    if not ret:
        break

    if frame_idx % skip == 0:
        _, frame = cap.retrieve()
        frames.append(frame)
        indices.append(frame_idx)

        if len(frames) >= BATCH_SIZE:
            results = model(frames, device=device, half=(device.startswith("cuda")))

            for res, fid in zip(results, indices):
                annotated = res.plot()
                cv2.imwrite(f"{DEBUG_DIR}/debug_frame{fid:06d}.png", annotated)

            frames, indices = [], []

    frame_idx += 1

if frames:
    results = model(frames, device=device, half=(device.startswith("cuda")))
    for res, fid in zip(results, indices):
        cv2.imwrite(f"{DEBUG_DIR}/debug_frame{fid:06d}.png", res.plot())

cap.release()

import os
import cv2
import torch
from ultralytics import YOLO
from multiprocessing import Process, Queue

VIDEO_PATH = "./assets/highlight.webm"
DEBUG_DIR  = "./debug"
SKIP_SEC   = 4
BATCH_SIZE = 16

EOS = None


def reader(frame_queue: Queue):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = int(fps * SKIP_SEC)
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % skip == 0:
            _, frame = cap.retrieve()
            frame_queue.put((idx, frame))
        idx += 1
    cap.release()
    frame_queue.put(EOS)


def inferencer(frame_queue: Queue, result_queue: Queue):
    model = YOLO("./best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device.startswith("cuda"):
        model.fuse()
        model.model.half()

    batch_idxs, batch_frames = [], []
    while True:
        item = frame_queue.get()
        if item is EOS:
            break

        fid, frame = item
        batch_idxs.append(fid)
        batch_frames.append(frame)

        if len(batch_frames) >= BATCH_SIZE or frame_queue.empty():
            results = model(batch_frames, device=device, half=(device.startswith("cuda")))
            for res, fid in zip(results, batch_idxs):
                annotated = res.plot()
                result_queue.put((fid, annotated))
            batch_idxs.clear()
            batch_frames.clear()

    result_queue.put(EOS)


def writer(result_queue: Queue):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    while True:
        item = result_queue.get()
        if item is EOS:
            break
        fid, annotated = item
        path = os.path.join(DEBUG_DIR, f"frame_{fid:06d}.png")
        cv2.imwrite(path, annotated)


def main():
    frame_q  = Queue(maxsize= BATCH_SIZE * 2)
    result_q = Queue(maxsize= BATCH_SIZE * 2)

    p1 = Process(target=reader,      args=(frame_q,))
    p2 = Process(target=inferencer,  args=(frame_q, result_q))
    p3 = Process(target=writer,      args=(result_q,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from multiprocessing import Process, Queue, shared_memory
from tqdm import tqdm
import onnxruntime as ort

VIDEO_PATH = "./assets/highlight.webm"
DEBUG_DIR  = "./debug"
ONNX_PATH  = "./best.onnx"
SKIP_SEC   = 4
BATCH_SIZE = 16
EOS        = None 

def get_resolution(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, total_frames

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = (int(w0*r), int(h0*r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw/2, dh/2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return im[np.newaxis, ...]

def reader(frame_q: Queue, shms, w, h, total_frames):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    skip = int(fps * SKIP_SEC)
    expected = (total_frames + skip - 1) // skip

    pb = tqdm(total=expected, desc="Reader", position=0)
    idx = 0
    shm_idx = 0

    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % skip == 0:
            _, frame = cap.retrieve()
            buf = np.ndarray((h, w, 3), np.uint8, buffer=shms[shm_idx].buf)
            buf[:] = frame
            frame_q.put((shm_idx, idx))
            shm_idx = (shm_idx + 1) % BATCH_SIZE
            pb.update(1)
        idx += 1

    cap.release()
    pb.close()
    frame_q.put(EOS)

def inferencer(frame_q: Queue, result_q: Queue, shms, w, h):
    desired = ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
    providers = [p for p in ort.get_available_providers() if p in desired] or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    inp_name = sess.get_inputs()[0].name
    IMG_SIZE = 640

    pb = tqdm(desc="Inferencer", position=1)
    batch_keys, batch_ids = [], []

    while True:
        item = frame_q.get()
        if item is EOS:
            break
        key, fid = item
        batch_keys.append(key)
        batch_ids.append(fid)

        if len(batch_keys) >= BATCH_SIZE:
            frames = [np.ndarray((h, w, 3), np.uint8, buffer=shms[k].buf)
                      for k in batch_keys]
            batch = np.vstack([letterbox(f, (IMG_SIZE, IMG_SIZE)) for f in frames])
            dets  = sess.run(None, {inp_name: batch})[0]

            for i, k, fid in zip(range(len(batch_keys)), batch_keys, batch_ids):
                dr = dets[i]
                dr = dr[dr[:, 4] > 0.25]
                buf = np.ndarray((h, w, 3), np.uint8, buffer=shms[k].buf)
                boxes = dr[:, :4].astype(int)
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(buf, (x1, y1), (x2, y2), (0,255,0), 2)
                    pb.update(1)
                result_q.put((k, fid))

            batch_keys.clear()
            batch_ids.clear()

    # flush leftovers
    if batch_keys:
        frames = [np.ndarray((h, w, 3), np.uint8, buffer=shms[k].buf)
                  for k in batch_keys]
        batch = np.vstack([letterbox(f, (IMG_SIZE, IMG_SIZE)) for f in frames])
        dets  = sess.run(None, {inp_name: batch})[0]

        for i, k, fid in zip(range(len(batch_keys)), batch_keys, batch_ids):
            dr = dets[i]
            dr = dr[dr[:, 4] > 0.25]
            buf = np.ndarray((h, w, 3), np.uint8, buffer=shms[k].buf)
            boxes = dr[:, :4].astype(int)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(buf, (x1, y1), (x2, y2), (0,255,0), 2)
                pb.update(1)
            result_q.put((k, fid))

    pb.close()
    result_q.put(EOS)

def writer(result_q: Queue, shms, w, h, total_frames):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    fps  = cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS)
    skip = int(fps * SKIP_SEC)
    expected = (total_frames + skip - 1) // skip

    pb = tqdm(total=expected, desc="Writer", position=2)
    while True:
        item = result_q.get()
        if item is EOS:
            break
        k, fid = item
        frame = np.ndarray((h, w, 3), np.uint8, buffer=shms[k].buf)
        cv2.imwrite(f"{DEBUG_DIR}/frame_{fid:06d}.png", frame)
        pb.update(1)
    pb.close()

def main():
    w, h, total = get_resolution(VIDEO_PATH)
    frame_bytes = w * h * 3
    shms = [shared_memory.SharedMemory(create=True, size=frame_bytes)
            for _ in range(BATCH_SIZE)]
    frame_q = Queue(maxsize=BATCH_SIZE*2)
    result_q= Queue(maxsize=BATCH_SIZE*2)

    p1 = Process(target=reader,     args=(frame_q, shms, w, h, total))
    p2 = Process(target=inferencer, args=(frame_q, result_q, shms, w, h))
    p3 = Process(target=writer,     args=(result_q, shms, w, h, total))

    for p in (p1, p2, p3): p.start()
    for p in (p1, p2, p3): p.join()

    for shm in shms:
        shm.close(); shm.unlink()

if __name__ == "__main__":
    main()

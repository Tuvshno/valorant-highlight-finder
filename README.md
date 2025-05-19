# Valorant Kill Finder

Medium Deep Learning Model and Pipeline that can identify your kills in any VOD.

![Highlightr Demo](docs/demo_resized.gif)  
Video from [Dacoit](https://www.youtube.com/@dacoiit). Adjusted to 1 frame/second.

## Model & Training

**Architecture & Framework**  
- **Base model**: YOLOv11m Object Detection
- **Framework**: YOLO Ultralytics + ONNX-export  

**Dataset**  
- **Total images**: 130  
  - Training: 113  
  - Validation: 12  
  - Test: 5  
- **Augmentations**  
  - Random saturation shift: −34% … +34%  

**Training Setup**  
- **Input resolution**: 640 × 640  
- **Batch size**: 8  
- **Epochs**: 100
- **Hardware**: NVIDIA GTX 1050 Ti, AMD Ryzen 9 5900X  
- **Total runtime**: ~8 hours  

**Performance**  
- **Precision**: 99.85 %  
- **Recall**: 100.00 %  
- **mAP@0.5**: 99.50 %  
- **mAP@0.5–0.95**: 91.36 %

> **Caution:** These near-perfect numbers were obtained on a tiny test set. The near-100 % precision and recall usually point to overfitting and insufficient data diversity.

I will soon add a more reliable benchmark once I find the time to annotate more data. However, the model still performs with high accuracy despite insufficient test set benchmarks.

![Training Metrics](docs/result_small.png)  
Training Metrics

## Inference Performance

Benchmarked the full pipeline on a 5 min 30 s (330 s) gameplay highlight using:

- **Hardware**: NVIDIA GTX 1050 Ti, AMD Ryzen 9 5900X  
- **Model input**: 640×640 RGB frames  
- **Sampling rate**: 0.5 fps (one frame every 2 s)

| Mode             | Total Time (s) |
|------------------|---------------:|
| Single-frame     | 80.81          |
| Batch (B=4)      | 76.90          |

> **Note:** Both modes processed the video in ≈80.8 which 4× faster than playback.  
> Batches on a more powerful card will likely show better speed-ups in batch mode.  

## Architecture

### 1. Decode & Sample Frames  
- **FFmpeg** reads your video (e.g. `.webm`)  
- Optionally apply an `fps` filter (e.g. 0.5 fps → one frame every 2 s)  
- Outputs raw RGB frames via a pipe

### 2. Preprocess  
- **Read** bytes → NumPy array 
- **Resize** to `640×640` per model specification
- **Normalize** pixels to `[0,1]` and reorder to `(1,3,640,640)` for the model

### 3. Inference  
- **ONNX Runtime** session with CUDA (GPU) + CPU fallback  
- `session.run(...)` on each single frame or a batch of `B` frames

### 4. Post-process  
- Convert raw output `(1,5,N)` → a list of `[x, y, w, h, conf]` boxes  
- **Filter** by confidence threshold (currently 0.85 -> leads to a few incorrect inferences. Best -> 0.87)  
- **Non-Max Suppression** (OpenCV) removes overlapping duplicates  

### 5. Annotate & Save  
- **Draw** green boxes and confidence labels on the original frame  
- **Write** each frame as a PNG (`./debug/...`)  

## Additional Features

- **Single-frame & Batch Inference**  
- **CUDA Support**  

## Development Process & Iterations

Over time I explored multiple approaches before reaching the final pipeline. Here’s the evolution:

1. **`process/template_matching.py`**  
   - Simple OpenCV template matching to locate the map that players would play on static frames.  
   - **Limitation:** HUD variations and very slow.

2. **`ocr_kills.py` → `ocr_batch.py` → `ocr_pipe.py` → `ocr_stream.py` → `ocr_pipe_multi.py`**  
   - Switched to Tesseract OCR to read the map/round/kill HUD.  
   - Built increasingly robust workflows:  
     - **Pipe mode** (`ocr_pipe.py`) piping raw frames via FFmpeg to OCR
     - **Batch mode** (`ocr_batch.py`) piping raw frames via FFmpeg in batches to OCR 
     - **Real-time** (`ocr_stream.py`) piping raw frames via FFmpeg to OCR (small adjustments)
     - **Multi-stage** (`ocr_pipe_multi.py`) parallelized reads + inference  
   - **Limitation:** OCR struggled on low contrast or anti-aliasing. Not very reliable.

At this point, I wasn't making any good progress using OCR to identify maps. So I decided to try to switch to a different UI to focus on. I still wasn't focused on kills specifically, but on other features that could help identify the kills.

3. **`round_detection.py`**  
   - Added a CV-based “round change” detector using frame differencing + contours.  
   - Helped segment gameplay into discrete rounds for focused analysis.
   - **Limitation:** Not very reliable.

After failing to make a good round finder with OCR, I tried to use audio matching to identify the kill sound.

4. **`audio_matchfinder.py`**  
   - Experimented with audio cue detection (gunshots, death sounds) via waveform matching.  
   - **Limitation:** Not very reliable.

Finally, I decided that I was focusing on the wrong things. I strictly focused on the UI of the kills and switched to a more robust method: deep learning.

5. **`dl_torch.py`**  
   - Moved to deep learning: trained a YOLOv11m model on custom annotated kill-counter images.
   - **3-process architecture**: Reader ⟶ Inferencer ⟶ Writer using `multiprocessing` + `SharedMemory`  
   - **Inference**: PyTorch model, manual batch collection & stacking  
   - **Pre/post-processing**: Custom `letterbox` padding, no NMS  
   - **Output**: Annotated PNG frames only  
   - **Limitation:** Convoluted pipeline and was difficult to work with and debug.


6. **Final Pipeline** (`dl_pipe.py` / ONNX version)  
   - **Single-process loop**: FFmpeg pipe → preprocess → ONNX Runtime → annotate → write  
   - **Decoding**: `ffmpeg-python` rawvideo pipe   
   - **Inference**: ONNX Runtime (CUDA+CPU) 
   - **Pre/post-processing**: OpenCV resize + normalize + NMS  
   - **Output**: Per-frame PNGs

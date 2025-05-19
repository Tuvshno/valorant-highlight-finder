import subprocess
import ffmpeg
import onnx
import onnxruntime as ort
import numpy as np
import logging
import cv2
import os
import time

from onnx import helper
from google.protobuf.json_format import MessageToDict

IN_VIDEO_PATH = "./assets/highlight.webm"
MODEL_PATH = "./best.onnx"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_onnx():
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    
    logger.info(f'Available EPs: {ort.get_available_providers()}')
    
    for _input in onnx_model.graph.input:
        elem_type = _input.type.tensor_type.elem_type
        np_dtype  = helper.tensor_dtype_to_np_dtype(elem_type)
        print(f"{_input.name}: {np_dtype}")
        
    for _input in onnx_model.graph.input:
        print(MessageToDict(_input))

    for _input in onnx_model.graph.input:
        dim = _input.type.tensor_type.shape.dim
        input_shape = [MessageToDict(d).get("dimValue") for d in dim]
        print(input_shape)

    for _output in onnx_model.graph.output:
        elem_type = _output.type.tensor_type.elem_type
        np_dtype  = helper.tensor_dtype_to_np_dtype(elem_type)
        print(f"{_output.name}: {np_dtype}")
    
    for _output in onnx_model.graph.output:
        shape = []
        for d in _output.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(d.dim_param)   
        print(f"{_output.name}: {shape}")
        
def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height
    
def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .filter('fps', fps=1/1)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def read_frame(process1, width, height, input_size=(640, 640)):
    logger.debug('Reading frame')
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        return None, None
    else:
        assert len(in_bytes) == frame_size
        
        full_res = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape((height, width, 3))  
            .copy()
        )
        
        resized = cv2.resize(full_res, input_size, interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        model_input = np.transpose(tensor, (2,0,1))[None, ...]

    return full_res, model_input

def read_frame_batch(process1, width, height, input_size=(640, 640)):
    logger.debug('Reading frame')
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        return None, None
    else:
        assert len(in_bytes) == frame_size
        
        full_res = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape((height, width, 3))  
            .copy()
        )
        
        resized = cv2.resize(full_res, input_size, interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        model_input = np.transpose(tensor, (2,0,1))
        

    return full_res, model_input

def process_frame(full_res, model_input, session, conf_threshold=0.85, nms_thresh=0.4):
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: model_input})[0]
    
    boxes5 = raw_output[0].transpose(1,0)
    
    h_full, w_full = full_res.shape[:2]
    scale_x, scale_y = w_full / 640.0, h_full / 640.0

    boxes = []
    confs = []
    for x_c, y_c, bw, bh, conf in boxes5:
        if conf < conf_threshold:
            continue

        x1 = ( x_c - bw / 2 ) * scale_x
        y1 = ( y_c - bh / 2 ) * scale_y
        x2 = ( x_c + bw / 2 ) * scale_x
        y2 = ( y_c + bh / 2 ) * scale_y
        
        boxes.append([x1, y1, x2-x1, y2-y1])
        confs.append(float(conf))
        
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_thresh)
    detections = 0
    for i in idxs:
        i = i[0] if isinstance(i, (tuple,list)) else i
        x, y, w_box, h_box = boxes[i]
        x, y, w_box, h_box = int(x), int(y), int(w_box), int(h_box)
        cv2.rectangle(full_res, (x,y), (x+w_box,y+h_box), (0,255,0), 2)
        cv2.putText(full_res, f"{confs[i]:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        detections += 1

    return full_res, detections
    
def process_frame_batch(batch_frames, batch_inputs, session, conf_threshold=0.85):
    input_name = session.get_inputs()[0].name
    batch_inputs_stacked = np.stack(batch_inputs, axis=0)
    raw_output = session.run(None, {input_name: batch_inputs_stacked})[0]
    
    annotated_batch = []
    for i, full_res in enumerate(batch_frames):
        boxes5 = raw_output[i].transpose(1,0)
        h_full, w_full = full_res.shape[:2]
        scale_x, scale_y = w_full / 640.0, h_full / 640.0

        for x_c, y_c, bw, bh, conf in boxes5:
            if conf < conf_threshold:
                continue

            x1 = x_c - bw / 2
            y1 = y_c - bh / 2
            x2 = x_c + bw / 2
            y2 = y_c + bh / 2

            pt1 = (int(x1 * scale_x), int(y1 * scale_y))
            pt2 = (int(x2 * scale_x), int(y2 * scale_y))

            cv2.rectangle(full_res, pt1, pt2, (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(full_res, label, (pt1[0], pt1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        annotated_batch.append(full_res)

    return annotated_batch

def write_frame(annotated, folder, frame_idx):
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./debug/{folder}/debug_frame{frame_idx:03d}.png', annotated)

def print_detections(detections_list, fps=0.5):
    for frame_idx, kill_count in detections_list:
        seconds = frame_idx / fps
        timestamp = time.strftime('%H:%M:%S', time.gmtime(seconds))
        print(f'Detection at {timestamp} : frame_dx {frame_idx} : {kill_count} kills ')

def run(in_filename, session):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)

    os.makedirs('./debug', exist_ok=True)    
    os.makedirs('./debug/single', exist_ok=True)
    frame_idx = 0
    detections_list = []
    while True:
        full_res, model_input = read_frame(process1, width, height)
        if full_res is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        annotated, detections = process_frame(full_res, model_input, session)

        write_frame(annotated, "single", frame_idx)
        if detections > 0:
            detections_list.append((frame_idx, detections))
        frame_idx += 1

    print_detections(detections_list)
    logger.info('Waiting for ffmpeg process1')
    process1.wait()
    logger.info('Done')
    
def run_batch(in_filename, session, batch_size):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    
    os.makedirs('./debug', exist_ok=True)    
    os.makedirs('./debug/batch', exist_ok=True)
    frame_idx = 0
    batch_frames, batch_inputs = [], []
    while True:
        full_res, model_input = read_frame_batch(process1, width, height)
        if full_res is None:
            logger.info('End of input stream')
            break

        batch_frames.append(full_res)
        batch_inputs.append(model_input)

        if len(batch_inputs) == batch_size:
            logger.debug('Processing batch')
            annotated_batch = process_frame_batch(batch_frames, batch_inputs, session)
            batch_frames, batch_inputs = [], []

            for _, annotated in enumerate(annotated_batch):
                write_frame(annotated, "batch", frame_idx)
                frame_idx += 1

    if len(batch_frames) != 0:
            logger.debug('Processing batch')
            annotated_batch = process_frame_batch(batch_frames, batch_inputs, session)
            for _, annotated in enumerate(annotated_batch):
                write_frame(annotated,"batch", frame_idx)
                frame_idx += 1
    
    logger.info('Waiting for ffmpeg process1')
    process1.wait()
    logger.info('Done')

def load_session():
    load_onnx()
    options = ort.SessionOptions()
    options.enable_profiling=True
    session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    return session

def run_performance_comparision(session):
    start_single = time.time()
    run(IN_VIDEO_PATH, session)
    end_single = time.time()
    
    start_batch = time.time()
    run_batch(IN_VIDEO_PATH, session, batch_size=2)
    end_batch = time.time()
    
    print(f'Single Inference: {end_single-start_single:03f}')
    print(f'Batch Inference: {end_batch-start_batch:03f}')

if __name__ == "__main__":
    session = load_session()
    run_performance_comparision(session)

    # run(IN_VIDEO_PATH, session)
    
import subprocess
import ffmpeg
import onnx
import onnxruntime as ort
import numpy as np
import logging
import cv2
import os

from onnx import helper
from google.protobuf.json_format import MessageToDict

IN_VIDEO_PATH = "./assets/highlight.webm"
OUT_VIDEO_PATH = "output.webm"
MODEL_PATH = "./best.onnx"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_onnx():
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    
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
        .filter('fps', fps=1/2)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

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

def process_frame(full_res, model_input, session, conf_threshold=0.9):
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: model_input})[0]
    
    boxes5 = raw_output[0].transpose(1,0)
    
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

    return full_res
    
def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def run(in_filename, out_filename, process_frame):
    
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    # process2 = start_ffmpeg_process2(out_filename, width, height)
    
    load_onnx()
    options = ort.SessionOptions()
    options.enable_profiling=True
    session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    os.makedirs('./debug', exist_ok=True)    
    frame_idx = 0
    while True:
        full_res, model_input = read_frame(process1, width, height)
        if full_res is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        annotated = process_frame(full_res, model_input, session)
        
        # write_frame(process2, annotated)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./debug/debug_frame{frame_idx:03d}.png', annotated)
        frame_idx += 1

    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    # logger.info('Waiting for ffmpeg process2')
    # process2.stdin.close()
    # process2.wait()

    logger.info('Done')

if __name__ == "__main__":
    run(IN_VIDEO_PATH, OUT_VIDEO_PATH, process_frame)
import cv2
import numpy as np
from tqdm import tqdm

def get_max_frames(video_path):
  """
  Returns the total number of frames from video file
  
  Args:
    video_path: Path to video file
    
  Returns:
    The total number of frames or -1 if the video cannot be opened
  """
  
  video = cv2.VideoCapture(video_path)
  if not video.isOpened():
    return -1
  
  total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
  video.release()
  return total_frames

video_path = 'clip_fracture.mp4'
image_path = 'match_found.png'
video = cv2.VideoCapture(video_path)
template = cv2.imread(image_path)
template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', template_grey)

skip_rate = video.get(cv2.CAP_PROP_FPS) / 10
print(skip_rate)
for i in tqdm(range(get_max_frames(video_path))):
  ret, frame = video.read()
  
  if not ret:
    print('No more frames')
    break
  
  if i % skip_rate != 0:
    continue
  
  frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  res = cv2.matchTemplate(frame_grey, template_grey, cv2.TM_SQDIFF)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  
  threshold = 0.5
  if max_val >= threshold:
    print(f'Detected at frame {i},  match value: {max_val}')

    cv2.imshow('Match Found', frame)
    key = cv2.waitKey(1)
    
video.release()
cv2.destroyAllWindows()


    
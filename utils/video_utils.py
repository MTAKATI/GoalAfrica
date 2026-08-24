import cv2
import numpy as np  # <-- Add this import

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0 or np.isnan(fps):
        fps = 24.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
# core/reader.py
from pathlib import Path
import time
from core.utils import LoadImages, LoadStreams
from core.features.speed_ocr import safe_distance_estimation
def video_reader_process(source, imgsz, stride, frame_queue, stop_event):
    source = str(source)
    is_file = Path(source).suffix[1:].lower() in [
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'dng', 'webp', 'mpo',
        'mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'
    ]
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
    frame_count = 0
    current_speed = 0
    safe_distance = 0
    try:
        for data in dataset:
            if stop_event.is_set():
                break
            frame_count += 1
            if webcam:
                path, im, im0s, vid_cap, s = data[0], data[1], data[2], data[3], data[4]
            else:
                path, im, im0s, vid_cap, s = data
            if frame_count == 1 or frame_count % 5 == 0:
                current_speed, safe_distance = safe_distance_estimation(im0s)
            while not stop_event.is_set():
                try:
                    frame_queue.put((path, im, im0s, s, current_speed, safe_distance, time.time()), timeout=0.05)
                    break
                except Exception:
                    continue
    finally:
        try:
            frame_queue.put(None, timeout=1.0)
        except Exception:
            pass

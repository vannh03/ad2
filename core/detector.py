# core/detector.py
import time
import torch
import numpy as np
from ultralytics import YOLO
from configs.settings import ROI_CONFIG, MODEL_DETECTION_PATH
from core.utils import scale_coords, select_device
from core.tracking import BYTETracker

class BYTETrackerArgs:
    def __init__(self):
        self.track_thresh = 0.25
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.frame_rate = 30
        self.min_box_area = 10
def detector_tracker_process(weights, imgsz, conf_thres, iou_thres,
                             frame_queue, detect_queue, stop_event, device=''):
    device_t = select_device(device)
    model_weight = str(weights[0]) if weights else str(MODEL_DETECTION_PATH)
    yolo_model = YOLO(model_weight)
    names = yolo_model.names if hasattr(yolo_model, 'names') and yolo_model.names else getattr(yolo_model.model, 'names', {})
    tracker_args = BYTETrackerArgs()
    tracker = BYTETracker(
        track_thresh=tracker_args.track_thresh,
        track_buffer=tracker_args.track_buffer,
        match_thresh=tracker_args.match_thresh,
        frame_rate=tracker_args.frame_rate,
        min_box_area=tracker_args.min_box_area,
        names=names
    )
    while not stop_event.is_set():
        try:
            data = frame_queue.get(timeout=0.5)
        except Exception:
            continue
        if data is None:
            detect_queue.put(None)
            break
        if len(data) >= 7:
            path, im, im0s, s, current_speed, safe_distance, t_capture = data[:7]
        else:
            path, im, im0s, s, current_speed, safe_distance = data[:6]
            t_capture = time.time()
        t_infer_start = time.time()
        t_capture = t_infer_start
        im_tensor = torch.from_numpy(im).to(device_t).float() / 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]
        results = yolo_model.predict(source=im_tensor, conf=conf_thres, iou=iou_thres, device=device_t, verbose=False)
        pred = results[0].boxes.data
        detections_for_distance = []
        im0 = im0s.copy()
        frame_width, frame_height = im0.shape[1], im0.shape[0]
        roi_r = ROI_CONFIG["right"]
        roi_l = ROI_CONFIG["left"]
        roi_x_min_right, roi_x_max_right = int(frame_width * roi_r["x_min"]), int(frame_width * roi_r["x_max"])
        roi_y_min_right, roi_y_max_right = int(frame_height * roi_r["y_min"]), int(frame_height * roi_r["y_max"])
        roi_x_min_left, roi_x_max_left = int(frame_width * roi_l["x_min"]), int(frame_width * roi_l["x_max"])
        roi_y_min_left, roi_y_max_left = int(frame_height * roi_l["y_min"]), int(frame_height * roi_l["y_max"])
        if len(pred):
            pred = pred.clone()
            pred[:, :4] = scale_coords(im_tensor.shape[2:], pred[:, :4], im0.shape).round()
            detections_for_tracker = []
            for *xyxy, conf, cls in reversed(pred):
                x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                conf_score = conf.item()
                class_id = int(cls.item())
                class_name = names[class_id]
                detections_for_tracker.append([x1, y1, x2, y2, conf_score])
                detections_for_distance.append((x1, y1, x2, y2, conf_score, class_name))
        else:
            detections_for_tracker = []
        if len(detections_for_tracker):
            det_arr = np.array(detections_for_tracker)
            bboxes = det_arr[:, :4]
            scores = det_arr[:, 4]
            class_ids = np.zeros(len(bboxes))
        else:
            bboxes, scores, class_ids = np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        tracked_objects = tracker.update(bboxes, scores, class_ids, im0)
        all_objects = {}
        for obj in tracked_objects:
            x, y, w, h = obj["tlwh"]
            centroid_x, centroid_y = x + w / 2, y + h / 2
            track_id = obj["track_id"]
            roi_side = None
            if roi_x_min_right <= centroid_x <= roi_x_max_right and roi_y_min_right <= centroid_y <= roi_y_max_right:
                roi_side = 'right'
            elif roi_x_min_left <= centroid_x <= roi_x_max_left and roi_y_min_left <= centroid_y <= roi_y_max_left:
                roi_side = 'left'
            if roi_side is not None:
                all_objects[track_id] = {"centroid": (centroid_x, centroid_y), "roi": roi_side}
        det_duration = time.time() - t_infer_start
        while not stop_event.is_set():
            try:
                detect_queue.put((path, im0, s, current_speed, safe_distance, tracked_objects, detections_for_distance, all_objects, t_capture, det_duration), timeout=0.2)
                break
            except Exception:
                continue

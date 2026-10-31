# core/decision.py
import time
import os
import platform
import threading
import torch
import torch.nn as nn
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import joblib
except Exception:
    joblib = None
import numpy as np
import cv2
from configs.settings import (
    MODEL_MLP_PATH,
    SCALER_PATH,
    STATUS_MAP,
    ROI_CONFIG,
    AUDIO_CONFIG,
    TSDA_EW_CONFIG
)
from core.features.angle import compute_angle
from core.features.distance import estimate_distance

class TSDAEWTracker:
    """
    Thuật toán TSDA-EW (Trajectory- and Safety-Distance-Aware Early-Warning)
    Thực thi theo chuẩn các Section 3 & 4 của bài báo hội nghị ICAA 2027:
      - O_i(t) = (x_i(t), y_i(t), d_i(t), v_i^rel(t), l_i(t), c_i)
      - R_d(i, t) = d_i(t) / (D_safe(t) + epsilon)
      - S_d(i, t) = max(0, 1 - R_d(i, t))
      - v_i^rel(t) = (d_i(t - Delta t) - d_i(t)) / Delta t
      - TTC_i(t) = d_i(t) / (v_i^rel(t) + epsilon) (khi v_i^rel > 0, ngược lại vô cực)
      - S_t(i, t) = min(1, tau_TTC / (TTC_i(t) + epsilon))
      - S_m(i, t) = 1 nếu v_i^rel(t) > v_min ngược lại 0
      - H_i(t) = w_d * S_d + w_t * S_t + w_m * S_m
      - P_i(t) = (1 / K) * sum(I[H_i(t-j) > tau_H])
      - W_i(t) = 1 khi H_i(t) > tau_H và P_i(t) >= eta và C_i(t) != normal
    """
    def __init__(self, config=None):
        cfg = config or TSDA_EW_CONFIG
        self.tau_ttc = cfg.get("tau_ttc", 3.0)
        self.tau_h = cfg.get("tau_h", 0.5)
        self.K = cfg.get("K", 5)
        self.eta = cfg.get("eta", 0.6)
        self.wd, self.wt, self.wm = cfg.get("weights", (0.4, 0.4, 0.2))
        self.v_min = cfg.get("v_min", 0.5)
        self.max_history_len = cfg.get("max_history_len", 10)
        self.distance_history = {}  # {obj_id: [(timestamp, distance)]}
        self.hazard_history = {}    # {obj_id: [H_i(t)]}

    def update(self, obj_id, current_distance, safe_distance, current_time):
        if current_distance <= 0.0:
            return {
                "v_rel": 0.0,
                "ttc": 99.9,
                "r_d": 1.0,
                "s_d": 0.0,
                "s_t": 0.0,
                "s_m": 0.0,
                "hazard_score": 0.0,
                "persistence": 0.0,
                "is_hazard": False
            }

        if obj_id not in self.distance_history:
            self.distance_history[obj_id] = []
            self.hazard_history[obj_id] = []

        self.distance_history[obj_id].append((current_time, current_distance))
        if len(self.distance_history[obj_id]) > self.max_history_len:
            self.distance_history[obj_id].pop(0)
        hist = self.distance_history[obj_id]
        t_start, d_start = hist[0]
        dt = current_time - t_start
        if dt > 0.05 and len(hist) >= 3:
            v_rel = (d_start - current_distance) / dt
            v_rel = max(-30.0, min(35.0, v_rel))
        else:
            v_rel = 0.0
        if v_rel > 0:
            ttc = current_distance / (v_rel + 1e-5)
        else:
            ttc = float("inf")
        d_safe = max(float(safe_distance), 5.0)  # Cự ly an toàn tối thiểu 5m
        r_d = current_distance / (d_safe + 1e-5)
        s_d = max(0.0, 1.0 - r_d)
        if ttc != float("inf"):
            s_t = min(1.0, self.tau_ttc / (ttc + 1e-5))
        else:
            s_t = 0.0
        s_m = 1.0 if v_rel > self.v_min else 0.0
        hazard_score = self.wd * s_d + self.wt * s_t + self.wm * s_m
        hazard_score = round(float(min(1.0, max(0.0, hazard_score))), 3)
        self.hazard_history[obj_id].append(hazard_score)
        if len(self.hazard_history[obj_id]) > self.K:
            self.hazard_history[obj_id].pop(0)
        recent_hazards = self.hazard_history[obj_id]
        p_i = sum(1 for h in recent_hazards if h > self.tau_h) / len(recent_hazards)
        is_hazard = (hazard_score > self.tau_h) and (p_i >= self.eta)
        return {
            "v_rel": round(float(v_rel), 2),
            "ttc": round(float(ttc), 1) if ttc != float("inf") else 99.9,
            "r_d": round(float(r_d), 2),
            "s_d": round(float(s_d), 3),
            "s_t": round(float(s_t), 3),
            "s_m": round(float(s_m), 1),
            "hazard_score": hazard_score,
            "persistence": round(float(p_i), 2),
            "is_hazard": is_hazard
        }

    def cleanup(self, stale_ids):
        """Dọn dẹp lịch sử của các ID đã rời khỏi khung hình"""
        for tid in stale_ids:
            self.distance_history.pop(tid, None)
            self.hazard_history.pop(tid, None)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, hidden_dim_2=32, hidden_dim_3=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_3, 3)
        )

    def forward(self, x):
        return self.net(x)

class FastMLPPreprocessor(object):
    """Bộ tiền xử lý NumPy thuần túy tương thích 100% với preprocessor.pkl (tránh xung đột phiên bản sklearn/joblib/pickle trên Python 3.6 Jetson Nano)."""
    NUMERIC_COLS = ("x", "dx", "y", "dy", "angle_sin", "angle_cos", "distance", "speed")
    MEAN = np.array([
        1233.9039939637828, -0.895521730382294, 977.0777665995977, 0.3802507042253521,
        0.8798874608384536, -0.035436811594276765, 5.239517102615694, 19.79678068410463
    ], dtype=np.float64)
    SCALE = np.array([
        563.4369580114202, 14.0677538760512, 68.82318914893169, 6.331145111567643,
        0.12976876092242284, 0.4557437408587742, 3.8257684711148836, 11.159577353035347
    ], dtype=np.float64)
    CATEGORIES = ("None", "left", "right")

    def transform(self, X):
        if isinstance(X, dict):
            rows = [X]
        elif isinstance(X, (list, tuple)):
            rows = X
        elif hasattr(X, "to_dict"):
            rows = X.to_dict("records")
        else:
            rows = list(X)
        out = np.zeros((len(rows), 11), dtype=np.float32)
        for i, r in enumerate(rows):
            vals = np.array([float(r.get(c, 0.0)) for c in self.NUMERIC_COLS], dtype=np.float64)
            out[i, :8] = ((vals - self.MEAN) / self.SCALE).astype(np.float32)
            roi_val = str(r.get("roi_side", "None"))
            for j, cat in enumerate(self.CATEGORIES):
                if roi_val == cat:
                    out[i, 8 + j] = 1.0
        return out


def load_mlp_model():
    if not MODEL_MLP_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {MODEL_MLP_PATH}. Vui lòng kiểm tra thư mục models/decision/."
        )
    preprocessor = FastMLPPreprocessor()
    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 11
    mlp_model = MLP(input_dim=input_dim)
    mlp_model.load_state_dict(torch.load(str(MODEL_MLP_PATH), map_location=device_torch))
    mlp_model.to(device_torch)
    mlp_model.eval()
    return preprocessor, mlp_model, device_torch

def play_audio_async(message):
    def _run():
        sys_name = platform.system()
        if sys_name == "Windows":
            try:
                import winsound
                # Tiếng bíp cảnh báo tần số 1200Hz trong 300ms
                winsound.Beep(1200, 300)
            except Exception:
                pass
        elif sys_name == "Darwin":
            try:
                # macOS Native TTS
                os.system(f'say -v Linh "{message}" > /dev/null 2>&1 || say "{message}" > /dev/null 2>&1')
            except Exception:
                pass
        else:
            try:
                os.system(f'espeak -v vi "{message}" > /dev/null 2>&1')
            except Exception:
                pass

    if AUDIO_CONFIG.get("enabled", True):
        t = threading.Thread(target=_run, daemon=True)
        t.start()

def feature_mlp_rule_process(detect_queue, output_queue, stop_event):
    preprocessor, mlp_model, device_torch = load_mlp_model()
    tsda_tracker = TSDAEWTracker()
    position_history, angle_history = {}, {}
    first_seen_x = {}
    last_seen = {}
    last_audio_time = 0
    audio_cooldown = AUDIO_CONFIG.get("cooldown_seconds", 5.0)
    current_frame_idx = 0
    last_valid_speed = 0.0
    while not stop_event.is_set():
        try:
            data = detect_queue.get(timeout=0.5)
        except Exception:
            continue

        if data is None:
            if output_queue is not None:
                output_queue.put(None)
            break

        current_frame_idx += 1
        t_start = time.time()
        if len(data) >= 10:
            path, im0, s, current_speed, safe_distance, tracked_objects, detections_for_distance, all_objects, t_capture, det_duration = data[:10]
        elif len(data) >= 9:
            path, im0, s, current_speed, safe_distance, tracked_objects, detections_for_distance, all_objects, t_capture = data[:9]
            det_duration = 0.02
        else:
            path, im0, s, current_speed, safe_distance, tracked_objects, detections_for_distance, all_objects = data[:8]
            t_capture = time.time()
            det_duration = 0.02
        if current_speed is not None and float(current_speed) > 0:
            last_valid_speed = float(current_speed)
        v_ego_kmh = float(current_speed) if (current_speed is not None and float(current_speed) > 0) else last_valid_speed
        v_ego_mps = v_ego_kmh / 3.6
        frame_width, frame_height = im0.shape[1], im0.shape[0]
        roi_r = ROI_CONFIG["right"]
        roi_l = ROI_CONFIG["left"]
        roi_x_min_right, roi_x_max_right = int(frame_width * roi_r["x_min"]), int(frame_width * roi_r["x_max"])
        roi_y_min_right, roi_y_max_right = int(frame_height * roi_r["y_min"]), int(frame_height * roi_r["y_max"])
        roi_x_min_left, roi_x_max_left = int(frame_width * roi_l["x_min"]), int(frame_width * roi_l["x_max"])
        roi_y_min_left, roi_y_max_left = int(frame_height * roi_l["y_min"]), int(frame_height * roi_l["y_max"])
        ego_position = (frame_width // 2, frame_height)
        crossing_detected = False
        wrong_way_detected = False
        telemetry_objects = []
        active_track_ids = set(all_objects.keys())

        for obj_id, obj_data in all_objects.items():
            last_seen[obj_id] = current_frame_idx
            centroid, roi_side = obj_data['centroid'], obj_data['roi']
            cv2.putText(im0, f"ID:{obj_id}", (int(centroid[0]) - 20, int(centroid[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            object_distance = 0.0
            for (x1, y1, x2, y2, conf, label) in detections_for_distance:
                if (x1 <= centroid[0] <= x2) and (y1 <= centroid[1] <= y2):
                    try:
                        object_distance = estimate_distance(
                            bbox=(x1, y1, x2, y2),
                            class_name=label,
                            frame_width=frame_width
                        )
                    except Exception:
                        object_distance = 0.0
                    break
            angle, angle_sin, angle_cos = compute_angle(centroid[0], centroid[1], ego_position)
            if obj_id not in position_history:
                position_history[obj_id], angle_history[obj_id] = [], []
                first_seen_x[obj_id] = centroid[0]

            position_history[obj_id].append((centroid[0], centroid[1]))
            angle_history[obj_id].append(angle)

            max_history = 10
            if len(position_history[obj_id]) > max_history:
                position_history[obj_id] = position_history[obj_id][-max_history:]
            if len(angle_history[obj_id]) > max_history:
                angle_history[obj_id] = angle_history[obj_id][-max_history:]

            x_change = y_change = angle_change = 0.0
            status_mlp = "normal"
            if len(position_history[obj_id]) >= max_history:
                x_coords = [p[0] for p in position_history[obj_id]]
                y_coords = [p[1] for p in position_history[obj_id]]
                angle_coords = angle_history[obj_id]
                x_change = x_coords[-1] - x_coords[0]
                y_change = y_coords[-1] - y_coords[0]
                angle_change = angle_coords[-1] - angle_coords[0]

                row_dict = {
                    "x": float(centroid[0]),
                    "dx": float(x_change),
                    "y": float(centroid[1]),
                    "dy": float(y_change),
                    "angle_sin": float(angle_sin),
                    "angle_cos": float(angle_cos),
                    "distance": float(object_distance),
                    "roi_side": roi_side if roi_side is not None else 'none',
                    "speed": float(current_speed if current_speed is not None else 0.0)
                }

                try:
                    if isinstance(preprocessor, FastMLPPreprocessor) or pd is None:
                        X_scaled = preprocessor.transform(row_dict)
                    else:
                        X_scaled = preprocessor.transform(pd.DataFrame([row_dict]))
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device_torch)
                    with torch.no_grad():
                        outputs = mlp_model(X_tensor)
                        pred_idx = int(outputs.argmax(1).cpu().numpy()[0])
                        status_mlp, mlp_color = STATUS_MAP.get(pred_idx, ("normal", (0, 255, 0)))
                except Exception:
                    status_mlp, mlp_color = "normal", (0, 255, 0)

            status_mlp = status_mlp.lower() if isinstance(status_mlp, str) else str(status_mlp)
            tsda_res = tsda_tracker.update(obj_id, object_distance, safe_distance, t_start)
            v_rel = tsda_res["v_rel"]
            ttc = tsda_res["ttc"]
            hazard_score = tsda_res["hazard_score"]
            persistence = tsda_res["persistence"]
            is_hazard = tsda_res["is_hazard"]
            center_x = frame_width / 2.0
            dist_to_center_now = abs(centroid[0] - center_x)
            dist_to_center_old = abs((centroid[0] - x_change) - center_x)
            lateral_inward = dist_to_center_old - dist_to_center_now
            is_overtaken_same_direction = (
                (roi_side == 'right') and
                (first_seen_x.get(obj_id, centroid[0]) < 0.78 * frame_width) and
                (x_change > 25.0)
            )

            is_crossing = False
            is_wrong_way = False
            if roi_side == 'left':
                # Lao từ trái sang phải vào tim đường: x_change > 10.0 và lateral_inward > 6.0
                if x_change > 10.0 and lateral_inward > 6.0 and abs(angle_change) > 4.0:
                    is_crossing = True
                elif status_mlp == "crossing" and x_change > 5.0 and lateral_inward > 3.0:
                    is_crossing = True

            elif roi_side == 'right':
                if x_change < -10.0 and lateral_inward > 6.0 and abs(angle_change) > 4.0:
                    is_crossing = True
                elif status_mlp == "crossing" and x_change < -5.0 and lateral_inward > 3.0:
                    is_crossing = True

                # B. Phát hiện xe đi ngược chiều (Wrong-Way) bên phải (KỂ CẢ Ở SÁT RÌA LỀ ĐƯỜNG):
                # Áp dụng cho mọi phương tiện ở làn bên phải và sát mép đường (không thuộc diện xe cùng chiều bị vượt)
                # Đang ở vùng đón đầu phía trước ca-pô (centroid[1] <= 0.72 * frame_height)
                # Lao nhanh trực diện đối đầu về phía xe chủ: y_change >= 8.0 px, v_rel >= 1.0 m/s, ttc <= 4.5s
                # Hoặc mô hình MLP nhận diện wrong-way/crossing kết hợp y_change >= 6.0 px, v_rel >= 0.8 m/s
                elif (centroid[1] <= 0.72 * frame_height) and not is_overtaken_same_direction:
                    if y_change >= 8.0 and v_rel >= 1.0 and ttc <= 4.5:
                        is_wrong_way = True
                    elif status_mlp in ('wrong-way', 'crossing') and y_change >= 6.0 and v_rel >= 0.8 and ttc <= 5.0:
                        is_wrong_way = True

            # 6. Quyết định cảnh báo cuối cùng W_i(t) theo thuật toán TSDA-EW (Equation 642):
            # W_i(t) = 1 khi và chỉ khi:
            # - Rủi ro được khẳng định qua thời gian (is_hazard: H_i > tau_h và P_i >= eta)
            # - VÀ sự kiện thực sự thuộc loại nguy hiểm (is_crossing hoặc is_wrong_way)
            # - VÀ cự ly nằm trong vùng va chạm tiềm tàng
            warning_triggered = False
            status = "normal"
            color = (0, 255, 0)

            if is_hazard and is_wrong_way and object_distance <= max(safe_distance, 15.0):
                status = "wrong-way"
                color = (0, 0, 255)
                wrong_way_detected = True
                warning_triggered = True
            elif is_hazard and is_crossing and object_distance <= max(safe_distance, 15.0):
                status = "crossing"
                color = (0, 255, 255)
                crossing_detected = True
                warning_triggered = True
            else:
                status = "normal"
                color = (0, 255, 0)
            obj_class = "vehicle"
            for (x1, y1, x2, y2, conf, label) in detections_for_distance:
                if (x1 <= centroid[0] <= x2) and (y1 <= centroid[1] <= y2):
                    obj_class = label
                    break

            telemetry_objects.append({
                "id": int(obj_id),
                "class": str(obj_class),
                "distance": float(object_distance),
                "status": str(status),
                "angle": round(float(angle), 1),
                "v_rel": float(v_rel),
                "ttc": float(ttc),
                "hazard_score": float(hazard_score),
                "persistence": float(persistence),
                "warning": bool(warning_triggered)
            })
            label_overlay = f"{status} {object_distance}m | TTC:{ttc}s | H:{hazard_score}"
            cv2.putText(im0, label_overlay, (int(centroid[0]) - 30, int(centroid[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.circle(im0, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        if current_frame_idx % 30 == 0:
            stale_ids = [tid for tid, frame_num in last_seen.items() if (current_frame_idx - frame_num) > 60]
            for tid in stale_ids:
                position_history.pop(tid, None)
                angle_history.pop(tid, None)
                first_seen_x.pop(tid, None)
                last_seen.pop(tid, None)
            tsda_tracker.cleanup(stale_ids)
        cv2.rectangle(im0, (roi_x_min_right, roi_y_min_right), (roi_x_max_right, roi_y_max_right), (0, 255, 0), 2)
        cv2.rectangle(im0, (roi_x_min_left, roi_y_min_left), (roi_x_max_left, roi_y_max_left), (0, 255, 0), 2)
        cv2.circle(im0, ego_position, 8, (0, 0, 255), -1)
        t_end = time.time()
        dec_duration = t_end - t_start
        total_ai_latency_ms = round((det_duration + dec_duration) * 1000, 1)
        fps = round(1.0 / max(0.001, (det_duration + dec_duration)), 1)
        speed_val = int(current_speed)
        safe_dist_val = round(float(safe_distance), 1)
        telemetry = {
            "fps": fps,
            "latency_ms": total_ai_latency_ms,
            "det_latency_ms": round(det_duration * 1000, 1),
            "dec_latency_ms": round(dec_duration * 1000, 1),
            "t_capture": t_capture,
            "current_speed": speed_val,
            "speed_kmh": speed_val,
            "safe_distance": safe_dist_val,
            "safe_dist_m": safe_dist_val,
            "objects": telemetry_objects,
            "crossing": crossing_detected,
            "wrong_way": wrong_way_detected,
            "warning_level": "CRITICAL" if wrong_way_detected else ("WARNING" if crossing_detected else "SAFE"),
            "warning_text": "Phát hiện xe đi ngược chiều" if wrong_way_detected else ("Phương tiện cắt ngang" if crossing_detected else "An toàn - Không có nguy cơ"),
            "frame_idx": current_frame_idx
        }
        current_time = time.time()
        if crossing_detected and (current_time - last_audio_time) > audio_cooldown:
            play_audio_async("Cảnh báo có phương tiện cắt ngang")
            last_audio_time = current_time
        elif wrong_way_detected and (current_time - last_audio_time) > audio_cooldown:
            play_audio_async("Cảnh báo phương tiện đi ngược chiều bên phải")
            last_audio_time = current_time
        if output_queue is not None:
            while not stop_event.is_set():
                try:
                    output_queue.put((path, im0, s, current_speed, safe_distance, telemetry), timeout=0.2)
                    break
                except Exception:
                    continue

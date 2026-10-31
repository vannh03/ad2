#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy ADAS một lệnh duy nhất (One-Click) dành cho Jetson Nano.
Người dùng chỉ cần gõ:
    python3 run.py
Hệ thống sẽ tự động cấu hình và đo đạc FPS & Latency trung bình.
"""
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess
_orig_subprocess_run = subprocess.run
def _patched_subprocess_run(*args, **kwargs):
    if 'capture_output' in kwargs:
        if kwargs.pop('capture_output'):
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE
    if 'text' in kwargs:
        kwargs['universal_newlines'] = kwargs.pop('text')
    return _orig_subprocess_run(*args, **kwargs)
subprocess.run = _patched_subprocess_run

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def check_env():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append(("PyTorch CUDA", "wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O /tmp/torch.whl && pip3 install /tmp/torch.whl && rm -f /tmp/torch.whl"))
    try:
        import cv2
    except ImportError:
        missing.append(("OpenCV", "sudo apt-get install -y python3-opencv"))
    try:
        import numpy
    except ImportError:
        missing.append(("NumPy", "sudo apt-get install -y python3-numpy"))
    try:
        import pandas
    except ImportError:
        missing.append(("Pandas", "sudo apt-get install -y python3-pandas"))
    try:
        import sklearn
    except ImportError:
        missing.append(("Scikit-learn", "sudo apt-get install -y python3-sklearn"))
    try:
        import joblib
    except ImportError:
        missing.append(("Joblib", "sudo apt-get install -y python3-joblib"))

    if missing:
        print("\n" + "=" * 65)
        print("⚠️ PHÁT HIỆN HỆ THỐNG JETSON NANO CHƯA CÓ ĐỦ CÁC GÓI NỀN TẢNG:")
        print("=" * 65)
        for name, cmd in missing:
            print(f" • Thiếu {name}:")
            print(f"   👉 Chạy lệnh: {cmd}\n")
        print("=" * 65)
        print("Sau khi chạy xong các lệnh trên, bạn chỉ cần gõ lại: python3 run.py")
        sys.exit(1)

check_env()

import torch
from configs.settings import MODEL_DETECTION_PATH, PROJECT_ROOT
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)

def main():
    print("=" * 65)
    print("🚗 HỆ THỐNG CẢNH BÁO VA CHẠM THÔNG MINH (ADAS) - JETSON NANO")
    print("=" * 65)

    # 1. Tự động kiểm tra phần cứng (GPU hay CPU)
    has_cuda = torch.cuda.is_available()
    device = "0" if has_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f" • Thiết bị tính toán : {device_name} (device='{device}')")

    # 2. Tự động tìm mô hình tốt nhất (.engine nếu có, không thì .pt)
    engine_path = ROOT / "models" / "detection" / "best.engine"
    pt_path = ROOT / "models" / "detection" / "best.pt"
    if engine_path.exists():
        weights_path = str(engine_path)
        model_type = "TensorRT FP16 Engine (Tối ưu GPU tối đa)"
    else:
        weights_path = str(pt_path)
        model_type = "PyTorch FP32/FP16 Model"
    print(f" • Mô hình YOLO       : {weights_path}")
    print(f"   └─ Định dạng       : {model_type}")

    # 3. Tự động tìm video mẫu trong thư mục
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        video_source = sys.argv[1]
    else:
        sample_videos = list(ROOT.glob("*.MP4")) + list(ROOT.glob("*.mp4"))
        if not sample_videos and (ROOT / "runs").exists():
            sample_videos = list((ROOT / "runs").rglob("*.mp4")) + list((ROOT / "runs").rglob("*.MP4"))
        if sample_videos:
            video_source = str(sample_videos[0])
        else:
            video_source = "0"  # Camera mặc định nếu không có video nào
    print(f" • Video kiểm thử     : {Path(video_source).name}")
    print(f" • Kích thước ảnh     : 320x320 pixels")
    print(f" • Số frame kiểm tra  : 100 frames (5 frame đầu khởi động bỏ qua)")
    print("=" * 65)
    print("⏳ Đang khởi động các tiến trình xử lý ngầm (Đợi 2-3 giây)...")

    frame_queue = Queue(maxsize=30)
    detect_queue = Queue(maxsize=30)
    output_queue = Queue(maxsize=30)
    stop_event = Event()

    processes = [
        Process(
            target=video_reader_process,
            args=(video_source, (320, 320), 32, frame_queue, stop_event),
            name="Run_VideoReader"
        ),
        Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=[weights_path],
                imgsz=(320, 320),
                conf_thres=0.25,
                iou_thres=0.45,
                device=device,
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=stop_event
            ),
            name="Run_DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="Run_DecisionEngine"
        )
    ]

    for p in processes:
        p.daemon = True
        p.start()

    fps_list = []
    ai_latency_list = []
    det_latency_list = []
    dec_latency_list = []
    e2e_latency_list = []

    frame_count = 0
    max_frames = 100
    warmup_frames = 5
    t_benchmark_start = None

    try:
        while True:
            try:
                data = output_queue.get(timeout=3.0)
            except Exception:
                if not processes[0].is_alive() and not processes[2].is_alive():
                    break
                continue

            if data is None:
                break

            frame_count += 1
            t_now = time.time()
            if t_benchmark_start is None:
                t_benchmark_start = t_now

            path, im0, s, current_speed, safe_distance, telemetry = data[:6]
            fps_val = telemetry.get('fps', 0.0)
            ai_lat = telemetry.get('latency_ms', 0.0)
            det_lat = telemetry.get('det_latency_ms', 0.0)
            dec_lat = telemetry.get('dec_latency_ms', 0.0)
            t_capture = telemetry.get('t_capture', t_now)
            e2e_lat = (t_now - t_capture) * 1000.0 if t_capture > 0 else ai_lat

            is_warmup = (frame_count <= warmup_frames)
            status_tag = "KHOI_DONG" if is_warmup else "CHUAN_XAC"

            if not is_warmup:
                fps_list.append(fps_val)
                ai_latency_list.append(ai_lat)
                det_latency_list.append(det_lat)
                dec_latency_list.append(dec_lat)
                e2e_latency_list.append(e2e_lat)

            # In thanh tiến trình cập nhật tại chỗ
            sys.stdout.write(
                f"\r[{status_tag}] Frame {frame_count:03d}/{max_frames} | "
                f"FPS: {fps_val:5.1f} | "
                f"Latency: {ai_lat:5.1f}ms (YOLO: {det_lat:4.1f}ms, Rule: {dec_lat:4.1f}ms) | "
                f"E2E: {e2e_lat:5.1f}ms"
            )
            sys.stdout.flush()

            if frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ Đã dừng sớm theo yêu cầu người dùng (Ctrl+C).")
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)

    print("\n" + "=" * 65)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC TRÊN JETSON NANO")
    print("=" * 65)

    if not fps_list:
        print("❌ Không thu thập đủ dữ liệu. Vui lòng kiểm tra lại nguồn video.")
        return

    n = len(fps_list)
    avg_fps = sum(fps_list) / n
    avg_ai_lat = sum(ai_latency_list) / n
    avg_det_lat = sum(det_latency_list) / n
    avg_dec_lat = sum(dec_latency_list) / n
    avg_e2e_lat = sum(e2e_latency_list) / n
    p95_lat = sorted(ai_latency_list)[int(0.95 * n)]

    print(f" • Tốc độ trung bình (Realtime FPS) : {avg_fps:.2f} FPS")
    print(f" • Độ trễ xử lý trung bình (Latency): {avg_ai_lat:.2f} ms")
    print(f"   ├─ Thời gian YOLO Detect & Track  : {avg_det_lat:.2f} ms")
    print(f"   └─ Thời gian Suy luận Luật + MLP : {avg_dec_lat:.2f} ms")
    print(f" • Độ trễ AI nhỏ nhất (Min Latency) : {min(ai_latency_list):.2f} ms")
    print(f" • Độ trễ AI lớn nhất (Max Latency) : {max(ai_latency_list):.2f} ms")
    print(f" • Độ trễ phân vị 95% (P95 Latency) : {p95_lat:.2f} ms")
    print(f" • Độ trễ toàn trình (End-to-End)   : {avg_e2e_lat:.2f} ms")
    print("=" * 65)
    print("🎯 ĐÁNH GIÁ TIÊU CHUẨN THỜI GIAN THỰC:")
    if avg_fps >= 20.0 and avg_ai_lat <= 50.0:
        print("  ✅ ĐẠT CHUẨN REALTIME TRÊN JETSON NANO (FPS >= 20, Latency < 50ms)")
    else:
        print(f"  ℹ️ Tốc độ hiện tại: {avg_fps:.1f} FPS (Khuyến nghị chuyển sang TensorRT để đạt >22 FPS)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

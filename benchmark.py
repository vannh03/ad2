import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Process, Queue, Event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from configs.settings import MODEL_DETECTION_PATH, PROJECT_ROOT
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)

def parse_opt():
    parser = argparse.ArgumentParser(description="ADAS Benchmark FPS & Latency (Jetson Nano / Edge Device)")
    # Tự động ưu tiên best.engine nếu tồn tại (TensorRT trên Jetson Nano)
    default_engine = ROOT / "models" / "detection" / "best.engine"
    default_weights = str(default_engine) if default_engine.exists() else str(MODEL_DETECTION_PATH)
    
    # Tìm video mẫu mặc định nếu có
    sample_videos = list(ROOT.glob("*.MP4")) + list(ROOT.glob("*.mp4"))
    if not sample_videos and (ROOT / "runs" / "mlp_dataset_videos").exists():
        sample_videos = list((ROOT / "runs" / "mlp_dataset_videos").glob("*.mp4"))
    default_source = str(sample_videos[0]) if sample_videos else "0"

    parser.add_argument('--source', type=str, default=default_source, help='đường dẫn video hoặc 0 cho webcam')
    parser.add_argument('--weights', type=str, default=default_weights, help='mô hình YOLO (.pt hoặc .engine)')
    parser.add_argument('--imgsz', type=int, default=320, help='kích thước ảnh inference (320 khuyến nghị cho Jetson)')
    parser.add_argument('--device', default='0', help='thiết bị: 0 cho GPU Jetson, hoặc cpu')
    parser.add_argument('--frames', type=int, default=100, help='số frame tối đa cần đo (0 để chạy hết video)')
    parser.add_argument('--warmup', type=int, default=5, help='số frame khởi động ban đầu bỏ qua (tránh lệch số liệu do load model)')
    return parser.parse_args()

def run_benchmark():
    opt = parse_opt()

    print("=" * 70)
    print("🚀 BẮT ĐẦU BENCHMARK HỆ THỐNG ADAS (FPS & LATENCY)")
    print("=" * 70)
    print(f" • Nguồn đầu vào  : {opt.source}")
    print(f" • Mô hình YOLO   : {opt.weights}")
    print(f" • Kích thước ảnh : {opt.imgsz}x{opt.imgsz}")
    print(f" • Thiết bị chạy  : {opt.device}")
    print(f" • Giới hạn frame : {opt.frames if opt.frames > 0 else 'Toàn bộ video'}")
    print(f" • Frame Warmup   : {opt.warmup} frames (bỏ qua khi tính trung bình)")
    print("=" * 70)

    frame_queue = Queue(maxsize=2)
    detect_queue = Queue(maxsize=2)
    output_queue = Queue(maxsize=2)
    stop_event = Event()

    processes = [
        Process(
            target=video_reader_process,
            args=(str(opt.source), (opt.imgsz, opt.imgsz), 32, frame_queue, stop_event),
            name="Bench_VideoReader"
        ),
        Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=[str(opt.weights)],
                imgsz=(opt.imgsz, opt.imgsz),
                conf_thres=0.25,
                iou_thres=0.45,
                device=opt.device,
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=stop_event
            ),
            name="Bench_DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="Bench_DecisionEngine"
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
    t_benchmark_start = None

    try:
        while True:
            try:
                data = output_queue.get(timeout=3.0)
            except Exception:
                # Nếu video reader và decision engine đã xong thì dừng
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

            is_warmup = (frame_count <= opt.warmup)
            status_tag = "WARMUP" if is_warmup else "VALID"

            if not is_warmup:
                fps_list.append(fps_val)
                ai_latency_list.append(ai_lat)
                det_latency_list.append(det_lat)
                dec_latency_list.append(dec_lat)
                e2e_latency_list.append(e2e_lat)

            # In cập nhật từng frame lên terminal
            sys.stdout.write(
                f"\r[{status_tag}] Frame {frame_count:03d} | "
                f"FPS: {fps_val:5.1f} | "
                f"AI Latency: {ai_lat:5.1f}ms (YOLO+Track: {det_lat:4.1f}ms, Rule+MLP: {dec_lat:4.1f}ms) | "
                f"E2E: {e2e_lat:5.1f}ms"
            )
            sys.stdout.flush()

            if opt.frames > 0 and frame_count >= opt.frames:
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ Đã nhận tín hiệu dừng từ người dùng (Ctrl+C). Đang tổng hợp số liệu...")
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ ĐO ĐẠC HIỆU NĂNG THỰC NGHIỆM")
    print("=" * 70)

    if not fps_list:
        print("❌ Không thu thập đủ frame hợp lệ sau giai đoạn Warmup.")
        return

    n_samples = len(fps_list)
    avg_fps = sum(fps_list) / n_samples
    avg_ai_lat = sum(ai_latency_list) / n_samples
    avg_det_lat = sum(det_latency_list) / n_samples
    avg_dec_lat = sum(dec_latency_list) / n_samples
    avg_e2e_lat = sum(e2e_latency_list) / n_samples

    min_ai_lat = min(ai_latency_list)
    max_ai_lat = max(ai_latency_list)
    p95_ai_lat = sorted(ai_latency_list)[int(0.95 * n_samples)]

    print(f" • Tổng số frame xử lý          : {frame_count} frames (Warmup: {min(opt.warmup, frame_count)}, Valid: {n_samples})")
    print(f" • Tốc độ xử lý trung bình (FPS) : {avg_fps:.2f} FPS")
    print(f" • Độ trễ AI trung bình (Mean)   : {avg_ai_lat:.2f} ms")
    print(f"   ├─ Phát hiện & Track (YOLO)  : {avg_det_lat:.2f} ms")
    print(f"   └─ Suy luận rủi ro (MLP+TSDA): {avg_dec_lat:.2f} ms")
    print(f" • Độ trễ AI nhỏ nhất (Min)      : {min_ai_lat:.2f} ms")
    print(f" • Độ trễ AI lớn nhất (Max)      : {max_ai_lat:.2f} ms")
    print(f" • Phân vị 95% độ trễ AI (P95)   : {p95_ai_lat:.2f} ms")
    print(f" • Độ trễ toàn trình End-to-End  : {avg_e2e_lat:.2f} ms")
    print("=" * 70)
    print("✅ Hoàn thành đo đạc!")

if __name__ == "__main__":
    run_benchmark()

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from multiprocessing import Process, Queue, Event
import time
import torch
from configs.settings import (
    MODEL_DETECTION_PATH,
    DEFAULT_VIDEO_SOURCE,
    DEFAULT_WEIGHTS,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    DEFAULT_CONF_THRES,
    DEFAULT_IOU_THRES,
)
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process,
    output_writer_process
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def parse_opt():
    default_weights = str(MODEL_DETECTION_PATH)
    if DEFAULT_WEIGHTS.lower() != "auto" and (ROOT / DEFAULT_WEIGHTS).exists():
        default_weights = str(ROOT / DEFAULT_WEIGHTS)
    elif (ROOT / "models" / "detection" / "best.engine").exists():
        default_weights = str(ROOT / "models" / "detection" / "best.engine")

    default_device = DEFAULT_DEVICE
    if default_device.lower() == "auto":
        default_device = "0" if torch.cuda.is_available() else "cpu"

    default_source = DEFAULT_VIDEO_SOURCE
    if default_source != "0" and not Path(default_source).is_absolute():
        candidate = ROOT / default_source
        if candidate.exists():
            default_source = str(candidate)

    parser = argparse.ArgumentParser(description="ADAS Collision Warning System - AIP491")
    parser.add_argument('--weights', nargs='+', type=str, default=[default_weights], help='model path(s)')
    parser.add_argument('--source', type=str, default=default_source, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[DEFAULT_IMGSZ], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=DEFAULT_CONF_THRES, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=DEFAULT_IOU_THRES, help='NMS IoU threshold')
    parser.add_argument('--device', default=default_device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results in opencv window')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    frame_queue = Queue(maxsize=2)
    detect_queue = Queue(maxsize=2)
    output_queue = Queue(maxsize=2)
    stop_event = Event()
    processes = [
        Process(
            target=video_reader_process,
            args=(opt.source, opt.imgsz, 32, frame_queue, stop_event),
            name="VideoReader"
        ),
        Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=opt.weights,
                imgsz=opt.imgsz,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                device=opt.device,
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=stop_event
            ),
            name="DetectorTracker"
        ),
        Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, stop_event),
            name="DecisionEngine"
        ),
        Process(
            target=output_writer_process,
            args=(output_queue, stop_event, opt.project, opt.name, opt.view_img, opt.nosave, opt.exist_ok),
            name="OutputWriter"
        )
    ]
    for p in processes:
        p.daemon = True
        p.start()
    print("\n" + "=" * 60)
    print("Hệ thống ADAS đang chạy...")
    print(f"Nguồn dữ liệu: {opt.source}")
    print(f"Model Detection: {opt.weights[0]}")
    print("Nhấn Ctrl + C hoặc phím 'q' trên cửa sổ để dừng hệ thống.")
    print("=" * 60 + "\n")
    try:
        while not stop_event.is_set():
            for p in processes:
                if not p.is_alive() and not stop_event.is_set():
                    pass
            time.sleep(0.5)
            if not processes[0].is_alive() and not processes[2].is_alive():
                break
    except KeyboardInterrupt:
        print("\n Đang dừng...")
        stop_event.set()
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1.0)
        print("Đã dừng.")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

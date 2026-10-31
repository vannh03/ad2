# test_pipeline_init.py
import sys
from pathlib import Path

# Đảm bảo root có trong sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("1. Kiểm tra configs...")
import configs.settings as s
print("   - PROJECT_ROOT:", s.PROJECT_ROOT)
print("   - MODEL_DETECTION_PATH:", s.MODEL_DETECTION_PATH, "-> Tồn tại:", s.MODEL_DETECTION_PATH.exists())
print("   - MODEL_MLP_PATH:", s.MODEL_MLP_PATH, "-> Tồn tại:", s.MODEL_MLP_PATH.exists())
print("   - SCALER_PATH:", s.SCALER_PATH, "-> Tồn tại:", s.SCALER_PATH.exists())

assert s.MODEL_DETECTION_PATH.exists(), "Không tìm thấy file model YOLO!"
assert s.MODEL_MLP_PATH.exists(), "Không tìm thấy file model MLP!"
assert s.SCALER_PATH.exists(), "Không tìm thấy file preprocessor scaler!"

print("\n2. Kiểm tra load mô hình MLP trong core/decision.py...")
from core.decision import load_mlp_model
preprocessor, mlp_model, device_torch = load_mlp_model()
print("   - Tải thành công MLP trên device:", device_torch)

print("\n3. Kiểm tra import các modules trong core...")
from core.features.distance import estimate_distance
from core.features.angle import compute_angle
from core.tracking import BYTETracker
print("   - Các core features và tracker import thành công!")

print("\n4. Kiểm tra tham số main.py...")
import subprocess
result = subprocess.run([sys.executable, "main.py", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
assert result.returncode == 0, f"Lỗi khi chạy main.py --help: {result.stderr}"
print("   - main.py --help chạy thành công!")

print("\nDone!")

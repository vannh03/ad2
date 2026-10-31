import contextlib
import functools
import importlib
import math
import os
import subprocess
import sys
import types
from pathlib import Path, PurePath

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Polyfill tương thích hoàn toàn cho Python 3.6.9 trên Jetson Nano ---
if not hasattr(functools, "cached_property"):
    class cached_property(object):
        def __init__(self, func):
            self.func = func
            self.attrname = func.__name__
            self.__doc__ = func.__doc__

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            val = self.func(instance)
            instance.__dict__[self.attrname] = val
            return val
    functools.cached_property = cached_property

_orig_lru_cache = functools.lru_cache
if not getattr(_orig_lru_cache, "_py36_patched", False):
    def _compat_lru_cache(maxsize=128, typed=False):
        if callable(maxsize) and isinstance(typed, bool):
            user_function = maxsize
            wrapper = _orig_lru_cache(maxsize=128, typed=typed)
            return wrapper(user_function)
        return _orig_lru_cache(maxsize=maxsize, typed=typed)
    _compat_lru_cache._py36_patched = True
    functools.lru_cache = _compat_lru_cache

_orig_subprocess_run = subprocess.run
if not getattr(_orig_subprocess_run, "_py36_patched", False):
    def _patched_subprocess_run(*args, **kwargs):
        if "capture_output" in kwargs:
            if kwargs.pop("capture_output"):
                kwargs["stdout"] = subprocess.PIPE
                kwargs["stderr"] = subprocess.PIPE
        if "text" in kwargs:
            kwargs["universal_newlines"] = kwargs.pop("text")
        return _orig_subprocess_run(*args, **kwargs)
    _patched_subprocess_run._py36_patched = True
    subprocess.run = _patched_subprocess_run

_orig_unlink = Path.unlink
if not getattr(_orig_unlink, "_py36_patched", False):
    def _compat_unlink(self, missing_ok=False):
        try:
            _orig_unlink(self)
        except FileNotFoundError:
            if not missing_ok:
                raise
    _compat_unlink._py36_patched = True
    Path.unlink = _compat_unlink

if not hasattr(PurePath, "is_relative_to"):
    def _is_relative_to(self, *other):
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False
    PurePath.is_relative_to = _is_relative_to

if not hasattr(contextlib, "nullcontext"):
    class _nullcontext(object):
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            pass
    contextlib.nullcontext = _nullcontext

if not hasattr(math, "prod"):
    def _math_prod(iterable, start=1):
        res = start
        for x in iterable:
            res *= x
        return res
    math.prod = _math_prod

try:
    from importlib import metadata as _im_meta
    if _im_meta is None:
        raise ImportError
except Exception:
    try:
        import importlib_metadata as _im_meta
    except Exception:
        class _FallbackMetadata(types.ModuleType):
            class PackageNotFoundError(Exception):
                pass
            @staticmethod
            def version(name):
                try:
                    import pkg_resources
                    return pkg_resources.get_distribution(name).version
                except Exception:
                    return "0.0.0"
            @staticmethod
            def distributions():
                return []
            @staticmethod
            def distribution(name):
                raise _FallbackMetadata.PackageNotFoundError(name)
        _im_meta = _FallbackMetadata("importlib.metadata")
    sys.modules["importlib.metadata"] = _im_meta
    setattr(importlib, "metadata", _im_meta)

ENV_FILE_PATH = PROJECT_ROOT / ".env"


def load_dotenv(dotenv_path=None):
    """Đọc file .env bằng Python thuần (không cần thư viện ngoài trên Jetson Nano)."""
    path = Path(dotenv_path) if dotenv_path else ENV_FILE_PATH
    if not path.exists():
        return False
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                elif "#" in val:
                    val = val.split("#", 1)[0].strip()
                if key:
                    os.environ.setdefault(key, val)
        return True
    except Exception:
        return False


ENV_LOADED = load_dotenv(ENV_FILE_PATH)


def _get_env_bool(key, default=False):
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_env_int(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_env_float(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


DEFAULT_VIDEO_SOURCE = os.environ.get("ADAS_VIDEO_SOURCE", "NO20250604-063730-010007F.MP4")
DEFAULT_WEIGHTS = os.environ.get("ADAS_WEIGHTS", "auto")
DEFAULT_IMGSZ = _get_env_int("ADAS_IMGSZ", 320)
DEFAULT_DEVICE = os.environ.get("ADAS_DEVICE", "auto")
DEFAULT_MAX_FRAMES = _get_env_int("ADAS_MAX_FRAMES", 100)
DEFAULT_WARMUP_FRAMES = _get_env_int("ADAS_WARMUP_FRAMES", 5)
DEFAULT_CONF_THRES = _get_env_float("ADAS_CONF_THRES", 0.25)
DEFAULT_IOU_THRES = _get_env_float("ADAS_IOU_THRES", 0.45)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_DETECTION_PATH = MODELS_DIR / "detection" / "best.pt"
MODEL_MLP_PATH = MODELS_DIR / "decision" / "mlp_model.pth"
SCALER_PATH = MODELS_DIR / "decision" / "preprocessor.pkl"
STATUS_MAP = {
    0: ("crossing", (0, 255, 255)),     # Màu vàng: Cắt ngang
    1: ("normal", (0, 255, 0)),         # Màu xanh lá: Bình thường
    2: ("wrong-way", (0, 0, 255))       # Màu đỏ: Đi ngược chiều
}
ROI_CONFIG = {
    "right": {
        "x_min": 0.50,
        "x_max": 0.95,
        "y_min": 0.10,
        "y_max": 0.70
    },
    "left": {
        "x_min": 0.05,
        "x_max": 0.4999,
        "y_min": 0.10,
        "y_max": 0.70
    }
}
DISTANCE_CONFIG = {
    "horizontal_fov": 130,  # Độ
    "real_heights": {
        "person": 1.6,
        "bicycle": 1.0,
        "motorcycle": 1.2,
        "car": 1.5,
        "bus": 3.2,
        "truck": 3.5,
    }
}
AUDIO_CONFIG = {
    "cooldown_seconds": _get_env_float("ADAS_AUDIO_COOLDOWN", 5.0),
    "enabled": _get_env_bool("ADAS_AUDIO_ENABLED", False)
}

# Cấu hình thuật toán suy luận rủi ro sớm TSDA-EW
TSDA_EW_CONFIG = {
    "tau_ttc": 3.0,               # Ngưỡng Time-To-Collision nguy cơ (giây)
    "tau_h": 0.5,                 # Ngưỡng Hazard Score kích hoạt cảnh báo
    "K": 5,                       # Kích thước cửa sổ trượt khẳng định thời gian (frames)
    "eta": 0.6,                   # Tỷ lệ khẳng định rủi ro (>= 3/5 frames)
    "weights": (0.4, 0.4, 0.2),   # Trọng số: (w_d khoảng cách, w_t TTC, w_m vận tốc)
    "v_min": 0.5,                 # Vận tốc tiếp cận tối thiểu có ý nghĩa (m/s)
    "max_history_len": 10         # Cửa sổ lịch sử khoảng cách để tính v_rel
}

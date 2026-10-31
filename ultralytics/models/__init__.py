from .yolo import YOLO, YOLOE, YOLOWorld
try:
    from .fastsam import FastSAM
    from .llm import LLM
    from .nas import NAS
    from .rtdetr import RTDETR
except Exception:
    FastSAM = LLM = NAS = RTDETR = None
__all__ = ('LLM', 'NAS', 'RTDETR', 'SAM', 'YOLO', 'YOLOE', 'FastSAM', 'YOLOWorld')

def __getattr__(name):
    """Lazy-import SAM so standard YOLO imports don't load optional torchvision internals."""
    if name == 'SAM':
        from .sam import SAM
        return SAM
    raise AttributeError(f'module {__name__} has no attribute {name}')
__version__ = '8.4.152'
import functools
import importlib
import os
import sys
from typing import TYPE_CHECKING

if not hasattr(functools, 'cached_property'):
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

if not os.environ.get('OMP_NUM_THREADS'):
    os.environ['OMP_NUM_THREADS'] = '1'
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
from ultralytics.models.yolo import YOLO, YOLOWorld, YOLOE
settings = SETTINGS
MODELS = ('YOLO', 'YOLOWorld', 'YOLOE', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'LLM')
PLATFORM_EXPORTS = ('Platform', 'AsyncPlatform', 'APIError', 'APIConnectionError')
__all__ = ('__version__', 'ASSETS', *MODELS, *(PLATFORM_EXPORTS if sys.version_info >= (3, 11) else ()), 'checks', 'download', 'settings')
if TYPE_CHECKING:
    from ultralytics.models import LLM, YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR
    from ultralytics_platform import APIConnectionError, APIError, AsyncPlatform, Platform

if sys.version_info < (3, 7):
    try:
        from ultralytics.models import NAS, RTDETR, FastSAM
    except Exception:
        pass

def __getattr__(name):
    """Lazy-import public classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module('ultralytics.models'), name)
    if name in PLATFORM_EXPORTS:
        if sys.version_info < (3, 11):
            raise ImportError('Ultralytics Platform requires Python 3.11 or newer.')
        return getattr(importlib.import_module('ultralytics_platform'), name)
    raise AttributeError(f'module {__name__} has no attribute {name}')

def __dir__():
    """Extend dir() to include lazily available public names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS) | set(PLATFORM_EXPORTS))
if __name__ == '__main__':
    print(__version__)
from functools import partial
from pathlib import Path
import torch
from ultralytics.cfg import TASKS
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .deep_oc_sort import DeepOCSORT
from .fast_tracker import FASTTracker
from .oc_sort import OCSORT
from .track_tracker import TRACKTRACK
TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT, 'tracktrack': TRACKTRACK, 'fasttrack': FASTTracker, 'ocsort': OCSORT, 'deepocsort': DeepOCSORT}

def on_predict_start(predictor, persist=False):
    """Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to reuse existing trackers if they are already attached.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    trackable = ('detect', 'segment', 'pose', 'obb')
    task = predictor.args.task
    if task in TASKS and task not in trackable:
        raise ValueError(f"❌ Task '{task}' doesn't support 'mode=track', valid tasks are {', '.join(trackable)}")
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))
    cfg.device = predictor.device
    if cfg.tracker_type not in TRACKER_MAP:
        raise AssertionError(f"Only {sorted(TRACKER_MAP)} are supported for now, but got '{cfg.tracker_type}'")
    predictor._feats = None
    if hasattr(predictor, '_hook'):
        predictor._hook.remove()
    if hasattr(predictor, '_orig_postprocess'):
        predictor.postprocess = predictor._orig_postprocess
        del predictor._orig_postprocess
    if cfg.tracker_type in {'botsort', 'tracktrack', 'deepocsort'} and cfg.with_reid and (cfg.model == 'auto'):
        from ultralytics.nn.modules.head import Detect
        if not (isinstance(predictor.model.model, torch.nn.Module) and isinstance(predictor.model.model.model[-1], Detect) and (not predictor.model.model.model[-1].end2end)):
            cfg.model = 'yolo26n-cls.pt'
        else:

            def pre_hook(module, input):
                predictor._feats = list(input[0])
            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
        trackers.append(tracker)
        if predictor.dataset.mode != 'stream':
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs
    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if hasattr(tracker_cls, 'setup_predictor'):
        tracker_cls.setup_predictor(predictor)

def on_predict_postprocess_end(predictor, persist=False):
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == 'obb'
    is_stream = predictor.dataset.mode == 'stream'
    tracker_cls = type(predictor.trackers[0])
    dets_del_list = tracker_cls.compute_frame_extras(predictor) if hasattr(tracker_cls, 'compute_frame_extras') else None
    for (i, result) in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path
        src = result.obb if is_obb else result.boxes
        det = src.cpu().numpy()
        kwargs = {'feats': getattr(result, 'feats', None)}
        if dets_del_list is not None:
            kwargs['dets_del'] = dets_del_list[i]
        tracks = tracker.update(det, result.orig_img, **kwargs)
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]
        update_args = {'obb' if is_obb else 'boxes': torch.as_tensor(tracks[:, :-1], device=src.data.device)}
        predictor.results[i].update(**update_args)

def register_tracker(model, persist):
    """Register or refresh the tracking callbacks on the model for object tracking during prediction.

    Any earlier registration is replaced in place, so repeat calls neither stack callbacks nor keep a stale `persist`.

    Args:
        model (object): The model to register tracking callbacks on, exposing a `callbacks` event mapping.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    for (event, fn) in (('on_predict_start', on_predict_start), ('on_predict_postprocess_end', on_predict_postprocess_end)):
        callbacks = model.callbacks[event]
        i = next((i for (i, cb) in enumerate(callbacks) if getattr(cb, 'func', None) is fn), None)
        if i is None:
            model.add_callback(event, partial(fn, persist=persist))
        else:
            callbacks[i] = partial(fn, persist=persist)
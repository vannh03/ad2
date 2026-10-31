import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from ultralytics import SETTINGS, __version__
from ultralytics.cfg import MODES, TASKS
from ultralytics.utils import ARGV, ENVIRONMENT, GIT, IS_PIP_PACKAGE, ONLINE, PYTHON_VERSION, RANK, TESTS_RUNNING, TORCH_VERSION
from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info, unwrap_model

def _post(url, data, timeout=5.0):
    """Send a one-shot JSON POST request."""
    try:
        body = json.dumps(data, separators=(',', ':')).encode()
        req = Request(url, data=body, headers={'Content-Type': 'application/json'})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass

def _arch(model):
    """Return the architecture a model is built from, i.e. 'yolo11n-seg', or None if it cannot be determined.

    The config travels inside a checkpoint, so fine-tuned models report the architecture they descend from however many
    generations back.
    """
    desc = f"{getattr(model, 'description', '')}".split()
    yaml = getattr(model, 'yaml', None) or getattr(getattr(model, 'model', None), 'yaml', None) or {}
    stem = Path(yaml.get('yaml_file', '')).stem or (desc[1] if len(desc) > 1 else '')
    return stem.lower()[:100] or None

class Events:
    """Collect and send anonymous usage analytics with rate-limiting.

    Event collection and transmission are enabled when sync is enabled in settings, the current process is rank -1 or 0,
    tests are not running, the environment is online, and the installation source is either pip or the official
    Ultralytics GitHub repository.

    Attributes:
        url (str): Measurement Protocol endpoint for receiving anonymous events.
        events (list[dict]): In-memory queue of event payloads awaiting transmission.
        rate_limit (float): Minimum time in seconds between POST requests.
        t (float): Timestamp of the last transmission in seconds since the epoch.
        metadata (dict): Static metadata describing runtime, installation source, and environment.
        enabled (bool): Flag indicating whether analytics collection is active.

    Methods:
        __init__: Initialize the event queue, rate limiter, and runtime metadata.
        __call__: Queue an event and trigger a non-blocking send when the rate limit elapses.
    """
    url = 'https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw'

    def __init__(self):
        """Initialize the Events instance with queue, rate limiter, and environment metadata."""
        self.events = []
        self.rate_limit = 30.0
        self.t = 0.0
        self.metadata = {'cli': Path(ARGV[0]).name == 'yolo', 'install': 'git' if GIT.is_repo else 'pip' if IS_PIP_PACKAGE else 'other', 'python': PYTHON_VERSION.rsplit('.', 1)[0], 'torch': TORCH_VERSION, 'CPU': get_cpu_info(), 'version': __version__, 'env': ENVIRONMENT, 'session_id': round(random.random() * 1000000000000000.0), 'engagement_time_msec': 1000}
        self.enabled = SETTINGS['sync'] and RANK in {-1, 0} and (not TESTS_RUNNING) and ONLINE and (IS_PIP_PACKAGE or GIT.origin == 'https://github.com/ultralytics/ultralytics.git')

    def __call__(self, cfg, device=None, run=None):
        """Queue an event and flush the queue asynchronously when the rate limit elapses.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
            run (BasePredictor | BaseTrainer, optional): The completed run, read for the mode's result fields.
        """
        if not self.enabled or cfg.mode not in MODES or cfg.task not in TASKS:
            return
        if len(self.events) < 25:
            params = {**self.metadata, 'task': cfg.task, 'model': Path(str(cfg.model)).name[:100] if cfg.model else None, 'device': str(device)}
            if cfg.mode == 'export':
                params['format'] = cfg.format
            elif cfg.mode == 'train':
                try:
                    params['data'] = Path(cfg.data).stem[:100] if isinstance(cfg.data, (str, Path)) else None
                    params['imgsz'] = cfg.imgsz
                    params['epochs_done'] = run.epoch + 1 - run.start_epoch
                    params['batch'] = run.batch_size
                    params['hours'] = round((time.time() - run.train_time_start) / 3600, 4)
                    params['n'] = len(run.train_loader.dataset)
                    if run.best_fitness is not None:
                        params['fitness'] = round(float(run.best_fitness), 5)
                    params['optimizer'] = type(run.optimizer).__name__
                    params['lr0'] = min((g['initial_lr'] for g in run.optimizer.param_groups))
                    flags = {'pretrained': bool(cfg.pretrained), 'cos_lr': cfg.cos_lr, 'amp': run.amp, 'rect': cfg.rect, 'multi_scale': bool(cfg.multi_scale), 'freeze': bool(cfg.freeze), 'dropout': cfg.dropout > 0, 'early_stop': run.epoch + 1 < run.epochs, 'resume': bool(cfg.resume)}
                    params['flags'] = ','.join((k for (k, v) in flags.items() if v)) or None
                    params['arch'] = _arch(unwrap_model(run.model))
                    params['ngpu'] = run.world_size if run.world_size > 1 else None
                    if device.type == 'cuda':
                        params['GPU'] = get_gpu_info(device.index or 0).rsplit(', ', 1)[0]
                except Exception:
                    pass
            elif cfg.mode in {'predict', 'track'}:
                try:
                    params['n'] = run.seen
                    params['pixels'] = run.pixels
                    for (k, v) in (run.speed or {}).items():
                        params[f'{k}_ms'] = round(v, 3)
                    params['batch'] = min(getattr(run.dataset, 'bs', 0), run.seen) or None
                    model = run.model
                    params['format'] = model.format
                    params['nc'] = len(getattr(model, 'names', None) or ()) or None
                    flags = {'compile': hasattr(model, '_orig_mod'), 'end2end': getattr(model, 'end2end', False), 'augment': cfg.augment}
                    params['flags'] = ','.join((k for (k, v) in flags.items() if v)) or None
                    params['arch'] = _arch(model)
                    meta = getattr(model, 'metadata', None) or {}
                    params['quantize'] = str(meta.get('args', {}).get('quantize') or cfg.quantize or 32)
                    if device.type == 'cuda':
                        params['GPU'] = get_gpu_info(device.index or 0).rsplit(', ', 1)[0]
                    session = getattr(model, 'session', None)
                    ov = getattr(model, 'ov_compiled_model', None)
                    devices = session.get_providers() if session else ov.get_property('EXECUTION_DEVICES') if ov else []
                    params['provider'] = devices[0] if devices else None
                except Exception:
                    pass
            params = dict([(k, v) for (k, v) in params.items() if v is not None][:25])
            self.events.append({'name': cfg.mode, 'params': params})
        t = time.time()
        if t - self.t < self.rate_limit:
            return
        payload_events = list(self.events)
        Thread(target=_post, args=(self.url, {'client_id': SETTINGS['uuid'], 'events': payload_events}), daemon=True).start()
        self.events = []
        self.t = t
events = Events()

def on_train_end(trainer):
    """Record an anonymous training event after final metrics are available."""
    events(trainer.args, trainer.device, trainer)

def on_val_start(validator):
    """Record an anonymous standalone validation event.

    A trainer's final validation retains mode=train, so the guard prevents a duplicate train event.
    """
    if validator.args.mode == 'val':
        events(validator.args, validator.device)

def on_predict_end(predictor):
    """Record an anonymous prediction event after per-image speeds are available."""
    events(predictor.args, predictor.device, predictor)

def on_export_start(exporter):
    """Record an anonymous export event."""
    events(exporter.args, exporter.device)
callbacks = {'on_train_end': on_train_end, 'on_val_start': on_val_start, 'on_predict_end': on_predict_end, 'on_export_start': on_export_start}
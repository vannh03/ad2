import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from ultralytics import __version__
from ultralytics.utils import ASSETS, DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, FLOAT_OR_INT, IS_VSCODE, LOGGER, PLATFORM_URL, RANK, ROOT, SETTINGS, SETTINGS_FILE, STR_OR_PATH, TESTS_RUNNING, YAML, IterableSimpleNamespace, checks, colorstr, deprecation_warn, vscode_msg
SOLUTION_MAP = {'count': 'ObjectCounter', 'crop': 'ObjectCropper', 'blur': 'ObjectBlurrer', 'workout': 'AIGym', 'heatmap': 'Heatmap', 'isegment': 'InstanceSegmentation', 'visioneye': 'VisionEye', 'speed': 'SpeedEstimator', 'queue': 'QueueManager', 'analytics': 'Analytics', 'inference': 'Inference', 'trackzone': 'TrackZone', 'region': 'RegionCounter', 'security': 'SecurityAlarm', 'parking': 'ParkingManagement', 'help': None}
MODES = ('train', 'val', 'predict', 'export', 'track', 'benchmark')
TASKS = ('detect', 'segment', 'semantic', 'depth', 'classify', 'pose', 'obb')
TASK2DATA = {'detect': 'coco8.yaml', 'segment': 'coco8-seg.yaml', 'semantic': 'cityscapes8.yaml', 'depth': 'depth8.yaml', 'classify': 'imagenet10', 'pose': 'coco8-pose.yaml', 'obb': 'dota8.yaml'}
TASK2CALIBRATIONDATA = {'detect': 'coco128.yaml', 'segment': 'coco128-seg.yaml', 'semantic': 'cityscapes8.yaml', 'depth': 'depth8.yaml', 'classify': 'imagenet100', 'pose': 'coco8-pose.yaml', 'obb': 'dota128.yaml'}
TASK2MODEL = {'detect': 'yolo26n.pt', 'segment': 'yolo26n-seg.pt', 'semantic': 'yolo26n-sem.pt', 'depth': 'yolo26n-depth.pt', 'classify': 'yolo26n-cls.pt', 'pose': 'yolo26n-pose.pt', 'obb': 'yolo26n-obb.pt'}
TASK2METRIC = {'detect': 'metrics/mAP50-95(B)', 'segment': 'metrics/mAP50-95(M)', 'semantic': 'metrics/mIoU', 'depth': 'metrics/delta1', 'classify': 'metrics/accuracy_top1', 'pose': 'metrics/mAP50-95(P)', 'obb': 'metrics/mAP50-95(B)'}
ARGV = sys.argv or ['', '']
_YOLO_CLI_COMMAND = [sys.executable, '-m', 'ultralytics.cfg.__init__']
SOLUTIONS_HELP_MSG = f"""\n    Arguments received: {['yolo', *ARGV[1:]]!s}. Ultralytics 'yolo solutions' usage overview:\n\n        yolo solutions SOLUTION ARGS\n\n        Where SOLUTION (optional) is one of {list(SOLUTION_MAP.keys())[:-1]}\n              ARGS (optional) are any number of custom 'arg=value' pairs like 'show_in=True' that override defaults\n                  at https://docs.ultralytics.com/usage/cfg\n\n    1. Call object counting solution\n        yolo solutions count source="path/to/video.mp4" region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"\n\n    2. Call heatmap solution\n        yolo solutions heatmap colormap=cv2.COLORMAP_PARULA model=yolo26n.pt\n\n    3. Call queue management solution\n        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" model=yolo26n.pt\n\n    4. Call workout monitoring solution for push-ups\n        yolo solutions workout model=yolo26n-pose.pt kpts=[6, 8, 10]\n\n    5. Generate analytical graphs\n        yolo solutions analytics analytics_type="pie"\n\n    6. Track objects within specific zones\n        yolo solutions trackzone source="path/to/video.mp4" region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"\n\n    7. Count objects inside specific regions\n        yolo solutions region source="path/to/video.mp4" region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"\n\n    8. Run security alarm monitoring (email alerts require Python API)\n        yolo solutions security source="path/to/video.mp4"\n\n    9. Monitor parking occupancy (create JSON annotations first via Python ParkingPtsSelection)\n        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json"\n\n    10. Streamlit real-time webcam inference GUI\n        yolo streamlit-predict\n    """
CLI_HELP_MSG = f"""\n    Arguments received: {['yolo', *ARGV[1:]]!s}. Ultralytics 'yolo' commands use the following syntax:\n\n        yolo TASK MODE ARGS\n\n        Where   TASK (optional) is one of {list(TASKS)}\n                MODE (required) is one of {list(MODES)}\n                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.\n                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'\n\n    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01\n        yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01\n\n    2. Predict a YouTube video using a pretrained segmentation model at image size 320:\n        yolo predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320\n\n    3. Validate a pretrained detection model at batch-size 1 and image size 640:\n        yolo val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640\n\n    4. Export a YOLO26n classification model to ONNX format at image size 224 by 128 (no TASK required)\n        yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128\n\n    5. Ultralytics solutions usage\n        yolo solutions count or any of {list(SOLUTION_MAP.keys())[1:-1]} source="path/to/video.mp4"\n\n    6. Run special commands:\n        yolo help\n        yolo checks\n        yolo version\n        yolo settings\n        yolo login API_KEY\n        yolo logout\n        yolo copy-cfg\n        yolo cfg\n        yolo solutions help\n\n    Docs: https://docs.ultralytics.com\n    Platform: https://platform.ultralytics.com\n    Community: https://community.ultralytics.com\n    GitHub: https://github.com/ultralytics/ultralytics\n    """
QUANTIZE_ALIASES = {'8': 8, '16': 16, '32': 32, 'int8': 8, 'fp16': 16, 'fp32': 32, 'w8a8': 8, 'w16a16': 16, 'w32a32': 32, 'w8a16': 'w8a16', 'w8a32': 'w8a32'}
QUANTIZE_DOCS_URL = 'https://docs.ultralytics.com/modes/export#quantization-options'
QUANTIZE_VALID_VALUES = "8, 16, 32, 'int8', 'fp16', 'fp32', 'w8a8', 'w16a16', 'w8a16', or 'w8a32'"
CFG_FLOAT_KEYS = frozenset({'warmup_epochs', 'box', 'cls', 'dfl', 'pose', 'kobj', 'rle', 'angle', 'dlog', 'dgrad', 'dis', 'degrees', 'shear', 'time', 'workspace', 'batch'})
CFG_FRACTION_KEYS = frozenset({'dropout', 'lr0', 'lrf', 'cls_pw', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'erasing', 'conf', 'iou', 'fraction', 'multi_scale', 'dlam'})
CFG_INT_KEYS = frozenset({'epochs', 'patience', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'nbs', 'save_period'})
CFG_INT_MIN = {'nbs': 1, 'max_det': 1, 'mask_ratio': 1, 'vid_stride': 1, 'seed': 0}
CFG_BOOL_KEYS = frozenset({'save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val', 'save_json', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop', 'save_frames', 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'show_boxes', 'keras', 'optimize', 'dynamic', 'simplify', 'nms', 'profile', 'channels_last', 'cls_remap'})
CFG_STR_KEYS = frozenset({'optimizer', 'split', 'copy_paste_mode', 'auto_augment'})

def cfg2dict(cfg):
    """Convert a configuration object to a dictionary.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted. Can be a file path, a string, a
            dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg=DEFAULT_CFG_DICT, overrides=None):
    """Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = _handle_deprecation(cfg2dict(cfg))
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}
    for k in ('project', 'name'):
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':
        cfg['name'] = Path(str(cfg.get('model') or '')).stem
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")
    check_cfg(cfg)
    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg, hard=True):
    """Check configuration argument types and values for the Ultralytics library.

    This function validates the types and values of configuration arguments, ensuring correctness and converting them if
    necessary. It checks for specific key types defined in global variables such as `CFG_FLOAT_KEYS`,
    `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`.

    Args:
        cfg (dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 0.937,  # valid float
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 0.937, 'save': True}

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys use [0.0, 1.0]; dataset fraction also accepts counts and [train, val, test] lists.
    """
    typed_keys = CFG_FLOAT_KEYS | CFG_FRACTION_KEYS | CFG_INT_KEYS | CFG_BOOL_KEYS | CFG_STR_KEYS | {'scale', 'compile'}
    for (k, v) in cfg.items():
        if v is None and (k == 'amp' or (DEFAULT_CFG_DICT.get(k) is not None and k in typed_keys and (k != 'auto_augment'))):
            raise TypeError(f"'{k}=None' is invalid. '{k}' must not be None.")
        if v is not None:
            if k in CFG_FLOAT_KEYS and (not isinstance(v, FLOAT_OR_INT)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                cfg[k] = float(v)
            elif k == 'scale':
                if isinstance(v, (list, tuple)):
                    if len(v) != 2 or not all((isinstance(x, (int, float)) for x in v)):
                        if hard:
                            raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int, float, or a tuple/list of two floats (i.e. '{k}=(0.5, 2.0)')")
                        continue
                    continue
                elif not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                    cfg[k] = v = float(v)
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_FRACTION_KEYS:
                if k == 'fraction' and isinstance(v, list):
                    if len(v) not in {2, 3} or not all(v[:2]) or (not all((type(x) is int and x >= 0 or (type(x) is float and 0.0 <= x <= 1.0) for x in v))):
                        raise ValueError(f"'{k}={v}' is invalid. Use [train, val] or [train, val, test] counts/ratios.")
                    cfg[k] = [float(x) if x in {0, 1} else x for x in v]
                    continue
                if k == 'fraction' and isinstance(v, bool):
                    raise TypeError(f"'{k}={v}' is of invalid type bool. Valid '{k}' types are int, float, or list")
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                    cfg[k] = v = float(v)
                valid = 0.0 <= v <= 1.0 or (k == 'fraction' and isinstance(v, int) and (v > 1))
                if not valid or (k == 'fraction' and v == 0.0):
                    raise ValueError(f"'{k}={v}' invalid. Use integer count >1 or ratio (0, 1] for fraction.")
                if k == 'fraction' and v == 1:
                    cfg[k] = 1.0
            elif k in CFG_INT_KEYS:
                if not isinstance(v, int):
                    if hard:
                        raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')")
                    cfg[k] = v = int(v)
                if k in CFG_INT_MIN and v < CFG_INT_MIN[k]:
                    raise ValueError(f"'{k}={v}' is an invalid value. '{k}' must be >= {CFG_INT_MIN[k]}.")
            elif k in CFG_BOOL_KEYS and (not isinstance(v, bool)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")
                cfg[k] = bool(v)
            elif k in CFG_STR_KEYS and (not isinstance(v, str)):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a str.")
                cfg[k] = str(v)
            elif k == 'compile' and (not isinstance(v, (bool, str))):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool or str (i.e. '{k}=True' or '{k}=max-autotune')")
                cfg[k] = bool(v)
            elif k == 'amp':
                if not isinstance(v, bool) and str(v).lower() not in {'fp16', 'bf16', 'fp32'}:
                    raise ValueError(f"'{k}={v}' is invalid. Valid '{k}' values are True, False, 'fp16', 'bf16', or 'fp32'.")
                cfg[k] = v.lower() if isinstance(v, str) else v
            elif k == 'quantize':
                scheme = QUANTIZE_ALIASES.get(str(v).lower())
                if scheme is None:
                    if hard:
                        raise ValueError(f"'{k}={v}' is invalid. Valid '{k}' values are {QUANTIZE_VALID_VALUES}. See {QUANTIZE_DOCS_URL}")
                else:
                    cfg[k] = scheme

def get_save_dir(args, name=None):
    """Return the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task', 'mode',
            and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name' or the
            'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", name="exp", task="detect", mode="train", exist_ok=True)
        >>> get_save_dir(args).parts[-3:]
        ('detect', 'my_project', 'exp')
    """
    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        project = args.project or ''
        if not Path(project).is_absolute():
            base = ROOT.parent / 'tests/tmp/runs' if TESTS_RUNNING else Path(SETTINGS['runs_dir'])
            worker = os.environ.get('PYTEST_XDIST_WORKER')
            if worker and TESTS_RUNNING:
                base = base / worker
            project = base / args.task / project
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)
    return Path(save_dir).resolve()

def _handle_deprecation(custom):
    """Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (dict): Configuration dictionary potentially containing deprecated keys.

    Returns:
        (dict): Updated configuration dictionary with deprecated keys replaced.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        {'show_boxes': True, 'show_labels': False, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    deprecated_mappings = {'boxes': ('show_boxes', lambda v: v), 'hide_labels': ('show_labels', lambda v: not bool(v)), 'hide_conf': ('show_conf', lambda v: not bool(v)), 'line_thickness': ('line_width', lambda v: v)}
    removed_keys = {'label_smoothing', 'save_hybrid', 'crop_fraction'}
    if 'end2end' in custom:
        end2end = custom.pop('end2end')
        if end2end is not None:
            if not isinstance(end2end, bool):
                raise TypeError("Deprecated 'end2end' must be a bool.")
            custom['nms'] = False if end2end else True if custom.get('nms') is True else None
            deprecation_warn(f'end2end={end2end}', f"nms={custom.get('nms')}")
    int8 = custom.pop('int8', None)
    half = custom.pop('half', None)
    if (int8 is not None or half is not None) and 'quantize' not in custom:
        int8_on = int8 is not None and str(int8).strip().lower() not in {'none', 'false', '0'}
        half_on = half is not None and str(half).strip().lower() not in {'none', 'false', '0'}
        custom['quantize'] = 8 if int8_on else 16 if half_on else None
        deprecation_warn('int8' if int8 is not None else 'half', 'quantize')
    for (old_key, (new_key, transform)) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))
    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)
    return custom

def check_dict_alignment(base, custom, e=None, allowed_custom_keys=None):
    """Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (dict): The base configuration dictionary containing valid keys.
        custom (dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.
        allowed_custom_keys (set | None): Optional set of additional keys that are allowed in the custom dictionary.

    Raises:
        SyntaxError: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SyntaxError:
        ...     print("Mismatched keys found")
        Mismatched keys found

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    (base_keys, custom_keys) = (frozenset(x.keys()) for x in (base, custom))
    if allowed_custom_keys is None:
        allowed_custom_keys = {'augmentations', 'save_dir'}
    mismatched = [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]
    if mismatched:
        from difflib import get_close_matches
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f'{k}={base[k]}' if base.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        LOGGER.info(CLI_HELP_MSG)
        raise SyntaxError(string) from e

def merge_equals_args(args):
    """Merge arguments around isolated '=' in a list of strings and join fragments with brackets.

    This function handles the following cases:
        1. ['arg', '=', 'val'] becomes ['arg=val']
        2. ['arg=', 'val'] becomes ['arg=val']
        3. ['arg', '=val'] becomes ['arg=val']
        4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']

    Args:
        args (list[str]): A list of strings where each element represents an argument or fragment.

    Returns:
        (list[str]): A list of strings where the arguments around isolated '=' are merged and fragments with brackets
            are joined.

    Examples:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_equals_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ''
    depth = 0
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '=' and 0 < i < len(args) - 1:
            new_args[-1] += f'={args[i + 1]}'
            i += 2
            continue
        elif arg.endswith('=') and i < len(args) - 1 and ('=' not in args[i + 1]):
            new_args.append(f'{arg}{args[i + 1]}')
            i += 2
            continue
        elif arg.startswith('=') and i > 0:
            new_args[-1] += arg
            i += 1
            continue
        depth += arg.count('[') - arg.count(']')
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ''
        i += 1
    if current:
        new_args.append(current)
    return new_args

def handle_yolo_login(args):
    """Log in to Ultralytics Platform with an API key or remove the saved key."""
    if args[0] == 'logout':
        SETTINGS['api_key'] = ''
        LOGGER.info("Logged out ✅. To log in again, use 'yolo login API_KEY'.")
        return
    api_key_url = f'{PLATFORM_URL}/settings?tab=api-keys'
    if len(args) < 2:
        LOGGER.info(f"Get an API key from {api_key_url} and then run 'yolo login API_KEY'.")
        return
    from ultralytics import APIConnectionError, APIError, Platform
    try:
        with Platform(api_key=args[1], base_url=PLATFORM_URL, timeout=30) as client:
            client.account.summary()
        SETTINGS['api_key'] = args[1]
        LOGGER.info('New authentication successful ✅')
    except APIError as error:
        raise SystemExit('Invalid API key' if error.status_code == 401 else f'Authentication failed (HTTP {error.status_code})') from None
    except APIConnectionError as error:
        raise SystemExit(f'Authentication request failed, check your connection: {error}') from None

def handle_yolo_settings(args):
    """Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset and updating individual settings. It should be
    called when executing a script with arguments related to YOLO settings management.

    Args:
        args (list[str]): A list of command line arguments for YOLO settings management.

    Examples:
        >>> handle_yolo_settings(["reset"])  # Reset YOLO settings
        >>> handle_yolo_settings(["runs_dir=path/to/dir"])  # Update a specific setting

    Notes:
        - If no arguments are provided, the function will display the current settings.
        - The 'reset' command will delete the existing settings file and create new default settings.
        - Other arguments are treated as key-value pairs to update specific settings.
        - The function will check for alignment between the provided settings and the existing ones.
        - After processing, the updated settings will be displayed.
        - For more information on handling YOLO settings, visit:
          https://docs.ultralytics.com/usage/settings
    """
    url = 'https://docs.ultralytics.com/usage/settings'
    try:
        if any(args):
            if args[0] == 'reset':
                SETTINGS_FILE.unlink()
                SETTINGS.reset()
                LOGGER.info('Settings reset successfully')
            else:
                new = dict((parse_key_value_pair(a) for a in args))
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for (k, v) in new.items():
                    LOGGER.info(f"✅ Updated '{k}={v}'")
        LOGGER.info(SETTINGS)
        LOGGER.info(f'💡 Learn more about Ultralytics Settings at {url}')
    except Exception as e:
        LOGGER.warning(f"settings error: '{e}'. Please see {url} for help.")

def handle_yolo_solutions(args):
    """Process YOLO solutions arguments and run the specified computer vision solutions pipeline.

    Args:
        args (list[str]): Command-line arguments for configuring and running the Ultralytics YOLO solutions.

    Examples:
        Run people counting solution with default settings:
        >>> handle_yolo_solutions(["count"])

        Run analytics with custom configuration:
        >>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video.mp4"])

        Run inference with custom configuration, requires Streamlit version 1.29.0 or higher.
        >>> handle_yolo_solutions(["inference", "model=yolo26n.pt"])

    Notes:
        - Arguments can be provided in the format 'key=value' or as boolean flags
        - Available solutions are defined in SOLUTION_MAP with their respective classes and methods
        - If an invalid solution is provided, defaults to 'count' solution
        - Output videos are saved in 'runs/solutions/exp' directory
        - For 'analytics' solution, frame numbers are tracked for generating analytical graphs
        - Video processing can be interrupted by pressing 'q'
        - Processes video frames sequentially and saves output in .avi format
        - If no source is specified, downloads and uses a default sample video
        - The inference solution will be launched using the 'streamlit run' command.
        - The Streamlit app file is located in the Ultralytics package directory.
    """
    from ultralytics.solutions.config import SolutionConfig
    full_args_dict = vars(SolutionConfig())
    overrides = {}
    for arg in merge_equals_args(args):
        arg = arg.lstrip('-').rstrip(',')
        if '=' in arg:
            try:
                (k, v) = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ''}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)
    if not args:
        LOGGER.warning("No solution name provided. i.e `yolo solutions count`. Defaulting to 'count'.")
        args = ['count']
    if args[0] == 'help':
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return
    elif args[0] in SOLUTION_MAP:
        solution_name = args.pop(0)
    else:
        LOGGER.warning(f"❌ '{args[0]}' is not a valid solution. 💡 Defaulting to 'count'.\n🚀 Available solutions: {', '.join(list(SOLUTION_MAP.keys())[:-1])}\n")
        solution_name = 'count'
    if solution_name == 'inference':
        checks.check_requirements('streamlit>=1.29.0')
        LOGGER.info('💡 Loading Ultralytics live inference app...')
        subprocess.run(['streamlit', 'run', str(ROOT / 'solutions/streamlit_inference.py'), '--server.headless', 'true', overrides.pop('model', 'yolo26n.pt')], check=False)
    else:
        import cv2
        from ultralytics import solutions
        solution = getattr(solutions, SOLUTION_MAP[solution_name])(is_cli=True, **overrides)
        cap = cv2.VideoCapture(solution.CFG['source'])
        if solution_name != 'crop':
            (w, h, fps) = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            if solution_name == 'analytics':
                (w, h) = (1280, 720)
            save_dir = get_save_dir(SimpleNamespace(task='solutions', name='exp', exist_ok=False, project=None))
            save_dir.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(save_dir / f'{solution_name}.avi'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        try:
            f_n = 0
            while cap.isOpened():
                (success, frame) = cap.read()
                if not success:
                    break
                f_n = f_n + 1
                results = solution(frame, f_n) if solution_name == 'analytics' else solution(frame)
                if solution_name != 'crop':
                    vw.write(results.plot_im)
                if solution.CFG['show'] and cv2.waitKey(1) & 255 == ord('q'):
                    break
        finally:
            cap.release()

def parse_key_value_pair(pair='key=value'):
    """Parse a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        key (str): The parsed key.
        value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo26n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo26n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    (k, v) = pair.split('=', 1)
    (k, v) = (k.strip(), v.strip())
    assert v, f"missing '{k}' value"
    return (k, smart_value(v))

def smart_value(v):
    """Convert a string representation of a value to its appropriate Python type.

    This function attempts to convert a given string into a Python object of the most appropriate type. It handles
    conversions to None, bool, int, float, and other types that can be evaluated safely.

    Args:
        v (str): The string representation of the value to be converted.

    Returns:
        (Any): The converted value. The type can be None, bool, int, float, or the original string if no conversion is
            applicable.

    Examples:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> print(smart_value("None"))
        None
        >>> smart_value("some_string")
        'some_string'

    Notes:
        - The function uses a case-insensitive comparison for boolean and None values.
        - For other types, it attempts to use Python's ast.literal_eval() function for safe evaluation.
        - If no conversion is possible, the original string is returned.
    """
    v_lower = v.lower()
    if v_lower == 'none':
        return None
    elif v_lower == 'true':
        return True
    elif v_lower == 'false':
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            (name, _, attr) = v.rpartition('.')
            module = sys.modules.get(name)
            if module and attr.isupper():
                value = getattr(module, attr, None)
                if isinstance(value, (int, float)):
                    return value
            return v

def entrypoint(debug=''):
    """Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and executing
    the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          Ultralytics documentation at https://docs.ultralytics.com.
    """
    args = (debug.split(' ') if debug else ARGV)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MSG)
        return
    special = {'checks': checks.collect_system_info, 'version': lambda : LOGGER.info(__version__), 'settings': lambda : handle_yolo_settings(args[1:]), 'cfg': lambda : YAML.print(DEFAULT_CFG_PATH), 'login': lambda : handle_yolo_login(args), 'logout': lambda : handle_yolo_login(args), 'copy-cfg': copy_default_cfg, 'solutions': lambda : handle_yolo_solutions(args[1:]), 'help': lambda : LOGGER.info(CLI_HELP_MSG)}
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}
    special.update({k[0]: v for (k, v) in special.items()})
    special.update({k[:-1]: v for (k, v) in special.items() if len(k) > 1 and k.endswith('s')})
    special = {**special, **{f'-{k}': v for (k, v) in special.items()}, **{f'--{k}': v for (k, v) in special.items()}}
    overrides = {}
    for a in merge_equals_args(args):
        if a.startswith('--'):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(','):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if '=' in a:
            try:
                (k, v) = parse_key_value_pair(a)
                if k == 'cfg' and v is not None:
                    LOGGER.info(f'Overriding {DEFAULT_CFG_PATH} with {v}')
                    overrides = {**{k: val for (k, val) in YAML.load(checks.check_yaml(v)).items() if k != 'cfg'}, **overrides}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ''}, e)
        elif a in TASKS:
            overrides['task'] = a
        elif a in MODES:
            overrides['mode'] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True
        elif a in {'half', 'int8', 'end2end', 'nms'}:
            overrides[a] = True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}")
        else:
            check_dict_alignment(full_args_dict, {a: ''})
    check_dict_alignment(full_args_dict, overrides)
    mode = overrides.get('mode')
    if mode is None:
        mode = DEFAULT_CFG.mode or 'predict'
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{CLI_HELP_MSG}")
    task = overrides.pop('task', None)
    if task:
        if task not in TASKS:
            if task == 'track':
                LOGGER.warning(f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}.")
                (task, mode) = ('detect', 'track')
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}")
        if 'model' not in overrides:
            overrides['model'] = TASK2MODEL[task]
    model = overrides.pop('model', DEFAULT_CFG.model)
    if model is None:
        model = 'yolo26n.pt'
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides['model'] = model
    stem = Path(model).stem.lower()
    if 'rtdetr' in stem:
        from ultralytics import RTDETR
        model = RTDETR(model)
    elif 'fastsam' in stem:
        from ultralytics import FastSAM
        model = FastSAM(model)
    elif 'sam_' in stem or 'sam2_' in stem or 'sam2.1_' in stem:
        from ultralytics import SAM
        model = SAM(model)
    else:
        from ultralytics import YOLO
        model = YOLO(model, task=task)
        if 'yoloe' in stem or 'world' in stem:
            cls_list = overrides.get('classes', DEFAULT_CFG.classes)
            if isinstance(cls_list, str):
                model.set_classes([c.strip() for c in overrides.pop('classes', cls_list).split(',')])
    if task != model.task:
        if task:
            LOGGER.warning(f"conflicting 'task={task}' passed with 'task={model.task}' model. Ignoring 'task={task}' and updating to 'task={model.task}' to match model.")
        task = model.task
    if mode in {'predict', 'track'} and 'source' not in overrides:
        overrides['source'] = 'https://ultralytics.com/images/boats.jpg' if task == 'obb' else DEFAULT_CFG.source or ASSETS
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {'train', 'val'}:
        if overrides.get('data') is None and (not overrides.get('resume')):
            overrides['data'] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == 'export':
        if 'format' not in overrides:
            overrides['format'] = DEFAULT_CFG.format or 'torchscript'
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")
    getattr(model, mode)(**overrides)
    LOGGER.info(f'💡 Learn more at https://docs.ultralytics.com/modes/{mode}')
    if IS_VSCODE and SETTINGS.get('vscode_msg', True):
        LOGGER.info(vscode_msg())

def copy_default_cfg():
    """Copy the default configuration file and create a new one with '_copy' appended to its name.

    This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it with '_copy'
    appended to its name in the current working directory. It provides a convenient way to create a custom configuration
    file based on the default settings.

    Examples:
        >>> copy_default_cfg()

    Notes:
        - The new configuration file is created in the current working directory.
        - After copying, the function prints a message with the new file's location and an example
          YOLO command demonstrating how to use the new configuration file.
        - This function is useful for users who want to modify the default configuration without
          altering the original file.
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(f"{DEFAULT_CFG_PATH} copied to {new_file}\nExample YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8")
if __name__ == '__main__':
    entrypoint(debug='')
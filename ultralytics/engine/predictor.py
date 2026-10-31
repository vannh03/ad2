"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo predict model=yolo26n.pt source=0                               # webcam
                                           img.jpg                         # image
                                           vid.mp4                         # video
                                           screen                          # screenshot
                                           path/                           # directory
                                           list.txt                        # list of images
                                           list.streams                    # list of streams
                                           'path/*.jpg'                    # glob
                                           'https://youtu.be/LNwODJXcvt4'  # YouTube
                                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (macOS-only)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # Sony IMX
                         yolo26n_rknn_model         # Rockchip RKNN
                         yolo26n_executorch_model   # PyTorch ExecuTorch
                         yolo26n_axelera_model      # Axelera AI
                         yolo26n_deepx_model        # DEEPX
                         yolo26n_qnn.onnx           # Qualcomm QNN
                         yolo26n.tflite             # LiteRT
                         yolo26n_ascend_model       # Huawei Ascend
"""
import platform
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.plotting import class_activation_map
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode
STREAM_WARNING = '\nInference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large\nsources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict for help.\n\nExample:\n    results = model(source=..., stream=True)  # generator of Results objects\n    for r in results:\n        boxes = r.boxes  # Boxes object for bbox outputs\n        masks = r.masks  # Masks object for segment masks outputs\n        probs = r.probs  # Class probabilities for classification outputs\n'

class BasePredictor:
    """A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference, and result
    processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (str | Path | None): Copy of args.data, the dataset YAML AutoBackend falls back to for class names.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        speed (dict[str, float] | None): Per-image preprocess, inference and postprocess times in ms, once run.
        pixels (int | None): Mean per-image inference area in pixels, once a run completes.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (Callable): Image transforms for classification.
        callbacks (dict[str, list[Callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the BasePredictor class.

        Args:
            cfg (str | Path | dict | SimpleNamespace): Path to a configuration file or a configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.speed = None
        self.pixels = None
        self.windows = []
        self.screen = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, already RGB and normalized to
                0.0-1.0, or [(H, W, 3) x N] for list of BGR uint8 arrays. See
                ultralytics.data.loaders.LoadTensor._single_check for tensor input requirements.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        if not isinstance(im, torch.Tensor):
            im = self.pre_transform(im)
            im = torch.from_numpy(im[0]).unsqueeze(0) if len(im) == 1 else torch.from_numpy(np.stack(im))
            im = im.to(self.device)
            im = im.permute(0, 3, 1, 2)
            if im.shape[1] == 3:
                im = im.flip(1)
            im = im.contiguous()
            im = (im.half() if self.model.fp16 else im.float()).div_(255)
        else:
            im = im.to(self.device)
            im = im.half() if self.model.fp16 else im.float()
        return im

    def inference(self, im, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        skip = self.source_type.tensor or self.args.augment or self.args.embed
        if self.args.visualize and getattr(self.model, 'base_model', True) and (not skip):
            return class_activation_map(self.model, im, self.batch[0], self.save_dir, *args, conf=self.args.conf, classes=self.args.classes, **kwargs)
        return self.model(im, *args, augment=self.args.augment, embed=self.args.embed, **kwargs)

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.args.rect and (self.model.format == 'pt' or (getattr(self.model, 'dynamic', False) and self.model.format != 'imx')), stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Perform inference on an image or stream.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            stream (bool): Whether to stream the inference results. If True, returns a generator.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Returns:
            (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))

    def predict_cli(self, source=None, model=None):
        """Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes the
        inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.

        Notes:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass

    def setup_source(self, source, stride=None):
        """Set up source and inference mode.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
                inference.
            stride (int, optional): Model stride for image size checking.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)
        self.dataset = load_inference_source(source=source, batch=self.args.batch, vid_stride=self.args.vid_stride, buffer=self.args.stream_buffer, channels=getattr(self.model, 'channels', 3))
        self.source_type = self.dataset.source_type
        if self.source_type.stream or self.source_type.screenshot or len(self.dataset) > 1000 or any(getattr(self.dataset, 'video_flag', [False])):
            import torchvision
            if not getattr(self, 'stream', True):
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Stream inference on input source and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info('')
        if self.model is None:
            self.setup_model(model)
        unsupported = [k for k in ('augment', 'embed', 'visualize') if getattr(self.args, k)]
        if not getattr(self.model, 'base_model', True) and unsupported:
            LOGGER.warning(f"{unsupported} not supported by this model (format='{self.model.format}'), ignoring.")
            (self.args.augment, self.args.embed, self.args.visualize) = (False, None, False)
        with self._lock:
            if self.model.format == 'pt' and self.model.end2end:
                self.model.model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
            self.setup_source(source if source is not None else self.args.source)
            if self.args.save or self.args.save_txt:
                (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            (self.seen, self.speed, self.pixels, self.windows, self.batch) = (0, None, None, [], None)
            px = 0
            profilers = (ops.Profile(device=self.device), ops.Profile(device=self.device), ops.Profile(device=self.device))
            self.run_callbacks('on_predict_start')
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks('on_predict_batch_start')
                (paths, im0s, s) = self.batch
                with profilers[0]:
                    im = self.preprocess(im0s)
                if not self.done_warmup:
                    self.model.warmup(im=im)
                    self.done_warmup = True
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from ([preds] if isinstance(preds, torch.Tensor) else preds)
                        continue
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks('on_predict_postprocess_end')
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        px += im.shape[2] * im.shape[3]
                        self.results[i].speed = {'preprocess': profilers[0].dt * 1000.0 / n, 'inference': profilers[1].dt * 1000.0 / n, 'postprocess': profilers[2].dt * 1000.0 / n}
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.save_crop or self.args.show:
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break
                if self.args.verbose:
                    LOGGER.info('\n'.join(s))
                self.run_callbacks('on_predict_batch_end')
                yield from self.results
            seen = self.seen
            if seen:
                t = tuple((x.t / seen * 1000.0 for x in profilers))
                self.speed = dict(zip(('preprocess', 'inference', 'postprocess'), t))
                self.pixels = round(px / seen)
                if self.args.verbose:
                    LOGGER.info(f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape {(min(self.args.batch, seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t)
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()
        if self.args.show:
            cv2.destroyAllWindows()
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks('on_predict_end')

    @smart_inference_mode(False)
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        model = deepcopy(model)
        self.model = AutoBackend(model=model or self.args.model, device=select_device(self.args.device, verbose=verbose), dnn=self.args.dnn, data=self.args.data, fp16=self.args.quantize == 16, channels_last=self.args.channels_last, fuse=True, verbose=verbose, end2end=self.args.nms is False)
        self.device = self.model.device
        if hasattr(self.model, 'imgsz') and (not getattr(self.model, 'dynamic', False)):
            self.args.imgsz = self.model.imgsz
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (list[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ''
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f'{i}: '
            frame = self.dataset.count
        else:
            match = re.search('frame (\\d+)/', s[i])
            frame = int(match[1]) if match else None
        self.txt_path = self.save_dir / 'labels' / (p.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))
        string += '{:g}x{:g} '.format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(line_width=self.args.line_width, boxes=self.args.show_boxes, conf=self.args.show_conf, labels=self.args.show_labels)
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)
        return string

    def save_predicted_images(self, save_path, frame=0):
        """Save video predictions as mp4/avi or images as jpg at specified path.

        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img
        if self.dataset.mode in {'stream', 'video'}:
            fps = self.dataset.fps if self.dataset.mode == 'video' else 30
            frames_path = self.save_dir / f'{save_path.stem}_frames'
            if save_path not in self.vid_writer:
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                (suffix, fourcc) = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                self.vid_writer[save_path] = cv2.VideoWriter(filename=str(Path(save_path).with_suffix(suffix)), fourcc=cv2.VideoWriter_fourcc(*fourcc), fps=fps, frameSize=(im.shape[1], im.shape[0]))
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f'{frames_path}/{save_path.stem}_{frame}.jpg', im)
        else:
            cv2.imwrite(str(save_path.with_suffix('.jpg')), im)

    def show(self, p=''):
        """Display an image in a window."""
        im = self.plotted_img
        if platform.system() in {'Linux', 'Windows'} and p not in self.windows:
            self.windows.append(p)
            name = p.encode('unicode_escape').decode()
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            (h, w) = im.shape[:2]
            try:
                if self.screen is None:
                    root = __import__('tkinter').Tk()
                    root.withdraw()
                    self.screen = (0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight())
                    root.destroy()
                r = min(self.screen[0] / w, self.screen[1] / h, 1.0)
                cv2.resizeWindow(name, max(1, int(w * r)), max(1, int(h * r)))
            except Exception:
                cv2.resizeWindow(name, w, h)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == 'image' else 1) & 255 == ord('q'):
            raise StopIteration

    def run_callbacks(self, event):
        """Run all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event, func):
        """Add a callback function for a specific event."""
        self.callbacks[event].append(func)
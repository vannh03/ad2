from dataclasses import dataclass, field
from typing import Any
import cv2

@dataclass
class SolutionConfig:
    """Manages configuration parameters for Ultralytics Vision AI solutions.

    The SolutionConfig class serves as a centralized configuration container for all the Ultralytics solution modules:
    https://docs.ultralytics.com/solutions#solutions. It leverages Python `dataclass` for clear, type-safe, and
    maintainable parameter definitions.

    Attributes:
        source (str, optional): Path to the input source (video, RTSP, etc.). Only usable with Solutions CLI.
        model (str, optional): Path to the Ultralytics YOLO model to be used for inference.
        classes (list[int], optional): List of class indices to filter detections.
        show_conf (bool): Whether to show confidence scores on the visual output.
        show_labels (bool): Whether to display class labels on visual output.
        show_boxes (bool): Whether to display bounding boxes on the visual output.
        region (list[tuple[int, int]], optional): Polygonal region or line for object counting.
        colormap (int, optional): OpenCV colormap constant for visual overlays (e.g., cv2.COLORMAP_DEEPGREEN).
        show_in (bool): Whether to display count number for objects entering the region.
        show_out (bool): Whether to display count number for objects leaving the region.
        up_angle (float): Upper angle threshold used in pose-based workouts monitoring.
        down_angle (int): Lower angle threshold used in pose-based workouts monitoring.
        kpts (list[int]): Keypoint indices to monitor, e.g., for pose analytics.
        analytics_type (str): Type of analytics to perform ("line", "area", "bar", "pie", etc.).
        figsize (tuple[float, float], optional): Size of the matplotlib figure used for analytical plots (width,
            height).
        blur_ratio (float): Ratio used to blur objects in the video frames (0.0 to 1.0).
        vision_point (tuple[int, int]): Reference point for directional tracking or perspective drawing.
        crop_dir (str): Directory path to save cropped detection images.
        json_file (str, optional): Path to a JSON file containing data for parking areas.
        line_width (int): Width for visual display, e.g. bounding boxes, keypoints, and counts.
        records (int): Number of detection records to send email alerts.
        fps (float): Frame rate (Frames Per Second) for speed estimation calculation.
        max_hist (int): Maximum number of historical points or states stored per tracked object for speed estimation.
        meter_per_pixel (float): Scale for real-world measurement, used in speed or distance calculations.
        max_speed (int): Maximum speed limit (e.g., km/h or mph) used in visual alerts or constraints.
        show (bool): Whether to display the visual output on screen.
        iou (float): Intersection-over-Union threshold for detection filtering.
        conf (float): Confidence threshold for keeping predictions.
        device (str, optional): Device to run inference on (e.g., 'cpu', '0' for CUDA GPU).
        max_det (int): Maximum number of detections allowed per video frame.
        quantize (int | str | None): Inference precision, e.g. 16 (FP16). Only PyTorch and TorchScript models compute in
            it; elsewhere the artifact and runtime decide, though 16 rounds OpenVINO inputs. Replaces the deprecated
            half flag.
        tracker (str): Path to tracking configuration YAML file (e.g., 'botsort.yaml').
        verbose (bool): Enable verbose logging output for debugging or diagnostics.
        data (str): Path to image directory used for similarity search.

    Methods:
        update: Update the configuration with user-defined keyword arguments and raise error on invalid keys.

    Examples:
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """
    source = None
    model = None
    classes = None
    show_conf = True
    show_labels = True
    show_boxes = True
    region = None
    colormap = cv2.COLORMAP_DEEPGREEN
    show_in = True
    show_out = True
    up_angle = 145.0
    down_angle = 90
    kpts = field(default_factory=lambda : [6, 8, 10])
    analytics_type = 'line'
    figsize = (12.8, 7.2)
    blur_ratio = 0.5
    vision_point = (20, 20)
    crop_dir = 'cropped-detections'
    json_file = None
    line_width = 2
    records = 5
    fps = 30.0
    max_hist = 5
    meter_per_pixel = 0.05
    max_speed = 120
    show = False
    iou = 0.7
    conf = 0.25
    device = None
    max_det = 300
    quantize = None
    imgsz = 640
    tracker = 'botsort.yaml'
    verbose = True
    data = 'images'

    def update(self, **kwargs):
        """Update configuration parameters with new values provided as keyword arguments."""
        if 'half' in kwargs:
            from ultralytics.utils import deprecation_warn
            deprecation_warn('half', 'quantize')
            kwargs['quantize'] = 16 if kwargs.pop('half') else None
        for (key, value) in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                url = 'https://docs.ultralytics.com/solutions#solutions-arguments'
                raise ValueError(f'{key} is not a valid solution argument, see {url}')
        return self
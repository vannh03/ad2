"""Depth estimation predictor for YOLO models."""
from typing import Any
import numpy as np
import torch
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops

class DepthPredictor(BasePredictor):
    """Predictor for YOLO depth estimation models.

    Produces per-pixel depth maps from RGB images.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthPredictor
        >>> predictor = DepthPredictor(overrides=dict(model="yolo26n-depth.pt"))
        >>> results = predictor("image.jpg")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DepthPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'depth'

    def postprocess(self, preds, img, orig_imgs):
        """Post-process depth predictions to Results objects."""
        depth_maps = preds[0] if isinstance(preds, (tuple, list)) else preds
        if depth_maps.ndim == 3:
            depth_maps = depth_maps.unsqueeze(1)
        depth_maps = ops.scale_masks(depth_maps, img.shape[2:], padding=False)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for (i, orig_img) in enumerate(orig_imgs):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            depth = ops.scale_masks(depth_maps[i:i + 1].float(), orig_img.shape[:2])
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, depth=depth.squeeze()))
        return results
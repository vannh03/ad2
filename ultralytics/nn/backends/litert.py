from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import LOGGER, NUM_THREADS
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend

class LiteRTBackend(BaseBackend):
    """Google LiteRT (formerly TensorFlow Lite) inference backend.

    Loads and runs inference with LiteRT models (.tflite files) exported via ai-edge-litert/litert-torch. Ultralytics
    exports keep float graph I/O (weights/activations may be int8/int16 internally); full-integer .tflite (legacy
    onnx2tf or third-party) with int8/int16 graph I/O are also handled by (de)quantizing at the boundary. Box and
    keypoint coordinates are denormalized by image size.
    """

    def load_model(self, weight):
        """Load a LiteRT model from a .tflite file.

        Args:
            weight (str | Path): Path to the .tflite model file (metadata embedded as a metadata.json entry).
        """
        check_requirements('ai-edge-litert>=2.1.4')
        from ai_edge_litert.interpreter import Interpreter
        tflite_file = Path(weight)
        LOGGER.info(f'Loading {tflite_file} for LiteRT inference...')
        self.interpreter = Interpreter(str(tflite_file), num_threads=NUM_THREADS)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.nhwc = self.input_details[0]['shape'][-1] == 3
        self.apply_metadata(self.read_metadata(tflite_file))

    def forward(self, im):
        """Run inference using the LiteRT interpreter.

        Box and pose keypoint coordinates are exported normalized to [0, 1] (so INT8 quantization preserves class
        score resolution) and denormalized here by the input image size, mirroring the TensorFlow Lite backend.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        im = im.cpu().numpy()
        if self.nhwc:
            im = im.transpose(0, 2, 3, 1)
        (h, w) = im.shape[1:3] if self.nhwc else im.shape[2:4]
        details = self.input_details[0]
        if details['dtype'] in {np.int8, np.int16}:
            (scale, zero_point) = details['quantization']
            im = (im / scale + zero_point).astype(details['dtype'])
        self.interpreter.set_tensor(details['index'], im)
        self.interpreter.invoke()
        kpt_start = 4 + len(self.names)
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output['index'])
            if self.task == 'semantic' and x.ndim == 3:
                y.append(x)
                continue
            if output['dtype'] in {np.int8, np.int16}:
                (scale, zero_point) = output['quantization']
                x = (x.astype(np.float32) - zero_point) * scale
            if x.ndim == 3 and (not self.end2end) and (getattr(self, 'head', None) != 'RTDETRDecoder'):
                x[:, [0, 2]] *= w
                x[:, [1, 3]] *= h
                if self.task == 'pose':
                    x[:, kpt_start::3] *= w
                    x[:, kpt_start + 1::3] *= h
            elif x.ndim == 3 and self.end2end and self.nhwc:
                x[:, :, [0, 2]] *= w
                x[:, :, [1, 3]] *= h
                if self.task == 'pose':
                    x[:, :, 6::3] *= w
                    x[:, :, 7::3] *= h
            y.append(x)
        if self.task == 'segment' and y[0].ndim == 4:
            y = [y[1], y[0]]
        if self.nhwc:
            if self.task == 'segment' and len(y) > 1 and (y[1].ndim == 4):
                y[1] = np.transpose(y[1], (0, 3, 1, 2))
            elif self.task in {'semantic', 'depth'} and len(y) == 1 and (y[0].ndim == 4):
                y[0] = np.transpose(y[0], (0, 3, 1, 2))
        return y
import json
from pathlib import Path
import torch
from ultralytics.utils import LOGGER, TORCH_VERSION

def torch2torchscript(model, im, output_file, metadata=None, prefix=''):
    """Export a PyTorch model to TorchScript format.

    Args:
        model (torch.nn.Module): The PyTorch model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor for tracing.
        output_file (Path | str): Path to save the exported TorchScript model.
        metadata (dict | None): Optional metadata to embed in the TorchScript archive.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``.torchscript`` file.
    """
    LOGGER.info(f'\n{prefix} starting export with torch {TORCH_VERSION}...')
    output_file = str(output_file)
    ts = torch.jit.trace(model, im, strict=False, check_trace=False)
    extra_files = {'config.txt': json.dumps(metadata or {})}
    ts.save(output_file, _extra_files=extra_files)
    return output_file
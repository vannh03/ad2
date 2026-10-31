from functools import partial
from pathlib import Path
import torch
from ultralytics.nn.modules import Pose, Pose26
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_executorch_requirements

def executorch_wrapper(model):
    """Apply ExecuTorch-specific model patches required for export/runtime compatibility."""
    import types
    for m in model.modules():
        if not isinstance(m, Pose):
            continue
        m.kpts_decode = types.MethodType(partial(_executorch_kpts_decode, is_pose26=type(m) is Pose26), m)
    return model

def _executorch_kpts_decode(self, kpts, is_pose26=False):
    """Decode pose keypoints for ExecuTorch export with XNNPACK-safe broadcasting."""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    y = kpts.view(bs, *self.kpt_shape, -1)
    anchors = self.anchors[None, None]
    strides = self.strides[None, None]
    a = (y[:, :, :2] + anchors if is_pose26 else y[:, :, :2] * 2.0 + (anchors - 0.5)) * strides
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)

def torch2executorch(model, im, output_dir, metadata=None, prefix=''):
    """Export a PyTorch model to ExecuTorch format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing/export.
        output_dir (Path | str): Directory to save the exported ExecuTorch model.
        metadata (dict | None, optional): Optional metadata to save as YAML.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported ExecuTorch model directory.
    """
    check_executorch_requirements()
    from executorch import version as executorch_version
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    LOGGER.info(f'\n{prefix} starting export with ExecuTorch {executorch_version.__version__}...')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pte_file = output_dir / 'model.pte'
    et_program = to_edge_transform_and_lower(torch.export.export(model, (im,)), partitioner=[XnnpackPartitioner()]).to_executorch()
    pte_file.write_bytes(et_program.buffer)
    if metadata is not None:
        YAML.save(output_dir / 'metadata.yaml', metadata)
    return str(output_dir)
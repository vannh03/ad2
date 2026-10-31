from pathlib import Path
import torch
from ultralytics.utils import ARM64, IS_JETSON, LOGGER, YAML

def torch2paddle(model, im, output_dir, metadata=None, prefix=''):
    """Export a PyTorch model to PaddlePaddle format using X2Paddle.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        output_dir (Path | str): Directory to save the exported PaddlePaddle model.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_paddle_model`` directory.
    """
    assert not IS_JETSON, 'Jetson Paddle exports not supported yet'
    from ultralytics.utils.checks import check_requirements
    check_requirements((('paddlepaddle-gpu>=3.0.0,<3.3.0' if im.device.type == 'cuda' else 'paddlepaddle==3.0.0' if ARM64 else 'paddlepaddle>=3.0.0,<3.3.0', 'paddlepaddle==3.0.0' if ARM64 else 'paddlepaddle>=3.0.0,<3.3.0', 'paddlepaddle-gpu>=3.0.0,<3.3.0'), 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle
    from x2paddle.op_mapper.pytorch2paddle import prim2code
    prim_assert = prim2code.prim_assert
    prim2code.prim_assert = lambda layer, **kwargs: None
    LOGGER.info(f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    try:
        pytorch2paddle(module=model, save_dir=output_dir, jit_type='trace', input_examples=[im])
    finally:
        prim2code.prim_assert = prim_assert
    if metadata:
        YAML.save(Path(output_dir) / 'metadata.yaml', metadata)
    return str(output_dir)
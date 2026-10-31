import json
from pathlib import Path
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements

def onnx2deepx(onnx_file, imgsz, dataset, metadata=None, optimize=False, prefix=''):
    """Convert an ONNX model to DEEPX format using the DEEPX DX-Compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        dataset (DataLoader): Calibration dataloader used to build the DEEPX config.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        optimize (bool, optional): If True, enables higher compiler optimization which reduces inference latency and
            increases compilation time. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (Path): Path to the exported DEEPX model directory.
    """
    try:
        import dx_com
    except ImportError:
        check_requirements('dx_com', cmds='-f https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html')
        import dx_com
    LOGGER.info(f'\n{prefix} starting export with DEEPX...')
    onnx_file = Path(onnx_file)
    export_path = onnx_file.parent / f'{onnx_file.stem}_deepx_model'
    export_path.mkdir(exist_ok=True)
    config_path = export_path / 'config.json'
    config = {'inputs': {'images': [1, 3, imgsz[0], imgsz[1]]}, 'calibration_num': 100, 'calibration_method': 'ema', 'default_loader': {'dataset_path': str(getattr(dataset.dataset, 'img_path', None) or dataset.dataset.root), 'file_extensions': [val for x in ['jpeg', 'jpg', 'png'] for val in (x.lower(), x.upper())], 'preprocessings': [{'resize': {'mode': 'pad', 'size': imgsz[0], 'pad_location': 'edge', 'pad_value': [114, 114, 114]}}, {'div': {'x': 255.0}}, {'convertColor': {'form': 'BGR2RGB'}}, {'transpose': {'axis': [2, 0, 1]}}, {'expandDim': {'axis': 0}}]}}
    with open(config_path, 'w') as file:
        json.dump(config, file)
    dx_com.compile(model=str(onnx_file), output_dir=str(export_path), config=str(config_path), opt_level=int(optimize))
    if metadata is not None:
        YAML.save(export_path / 'metadata.yaml', metadata)
    return export_path
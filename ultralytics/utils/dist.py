import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from . import USER_CONFIG_DIR
from .patches import torch_save
from .torch_utils import TORCH_1_9
if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

def find_free_network_port():
    """Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.

    Notes:
        Candidates are drawn below the default OS ephemeral floor (32768 on Linux, 49152 on macOS and Windows)
        because the port is released here and rebound later by the DDP subprocess. An ephemeral port can be handed to
        any outbound connection in that window, which surfaces as an EADDRINUSE rendezvous failure at launch.
    """
    import random
    import socket
    for port in random.SystemRandom().sample(range(10000, 32768), 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def generate_ddp_file(trainer):
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs. The file
    contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing training configuration and arguments.
            Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer class and callback reconstruction
        - Configuration overrides from the trainer arguments
        - Training initialization code
    """
    import cloudpickle
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_', suffix=f'{id(trainer)}.py', mode='w+', encoding='utf-8', dir=USER_CONFIG_DIR / 'DDP', delete=False) as file:
        path = Path(file.name).with_suffix('.pt')
        torch_save({'trainer': type(trainer), 'args': vars(trainer.args), 'model': trainer.model, 'callbacks': trainer.callbacks}, path, pickle_module=cloudpickle)
        file.write(f"""\n# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)\nif __name__ == "__main__":\n    import sys\n    sys.path = {sys.path!r}\n\n    from ultralytics.utils import DEFAULT_CFG_DICT\n    from ultralytics.utils.patches import torch_load\n\n    state = torch_load({str(path)!r}, map_location="cpu")\n\n    cfg = DEFAULT_CFG_DICT.copy()\n    cfg.update(save_dir='')   # handle the extra key 'save_dir'\n    trainer = state["trainer"](cfg=cfg, overrides=state["args"], _callbacks=state["callbacks"])\n    trainer.model = state["model"]\n    trainer.train()\n""")
    return file.name

def generate_ddp_command(trainer):
    """Generate command for distributed training.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)
    file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{trainer.world_size}', '--master_port', f'{port}', file]
    return (cmd, file)

def ddp_cleanup(trainer, file):
    """Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created as a
    temporary file for DDP training, and deletes it if so.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer used for distributed training.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f'{id(trainer)}.py' in file:
        os.remove(file)
        Path(file).with_suffix('.pt').unlink(missing_ok=True)
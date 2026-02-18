import os
import subprocess
import sys
from typing import Optional


def train_segmentor(config: str,
                    gpus: int = 1,
                    work_dir: Optional[str] = None,
                    extra_args: Optional[list] = None):
    """Launch training via the native Jittor training entry."""
    cmd = [
        sys.executable,
        os.path.join('utils', 'train.py'),
        f'--config={config}',
        f'--gpus={gpus}',
    ]
    if work_dir is not None:
        cmd.append(f'--save_path={work_dir}')
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


"""DynamiCrafter backbone construction from local source and checkpoints."""
import importlib
from pathlib import Path
import sys

from .checkpoints import load_state_dict_file


def load_dynamicrafter(source_root, checkpoint, config_path, device):
    from omegaconf import OmegaConf

    root = Path(source_root).resolve()
    if not (root / 'lvdm').is_dir():
        raise FileNotFoundError(f'DynamiCrafter source not found: {root}')
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f'DynamiCrafter checkpoint not found: {checkpoint}')
    settings = OmegaConf.load(config_path)
    base_config = root / settings.backbone.config
    config = OmegaConf.load(base_config).model
    # The full backbone checkpoint contains both OpenCLIP encoders.
    config.params.cond_stage_config.params.version = None
    config.params.img_cond_stage_config.params.version = None
    config.params.cond_stage_config.params.device = str(device)
    config.params.img_cond_stage_config.params.device = str(device)
    if str(root) not in sys.path:
        sys.path.append(str(root))
    utility_package = importlib.import_module('utils')
    upstream_utils = str(root / 'utils')
    if upstream_utils not in utility_package.__path__:
        utility_package.__path__.append(upstream_utils)
    module_name, class_name = config.target.rsplit('.', 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    backbone = constructor(**OmegaConf.to_container(config.params, resolve=True))
    state = load_state_dict_file(checkpoint)
    state = {key.replace('framestride_embed', 'fps_embedding'): value for key, value in state.items()}
    backbone.load_state_dict(state, strict=True)
    return backbone.to(device)

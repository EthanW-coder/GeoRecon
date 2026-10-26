"""Local checkpoint loading utilities."""
import inspect
from pathlib import Path
import torch


def load_state_dict_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a checkpoint dictionary')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise ValueError('Expected a model state dictionary')
    return state

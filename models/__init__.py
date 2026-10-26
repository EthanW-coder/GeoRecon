"""Scene reconstruction and video generation models."""
from importlib import import_module

_EXPORTS = {
    'RenderConfig': 'depmask_renderer',
    'ConfidenceEstimator': 'depmask_renderer',
    'DepthEdgeDetector': 'depmask_renderer',
    'PointCloudRenderer': 'depmask_renderer',
    'DepMaskRenderPipeline': 'depmask_renderer',
    'VideoDiffusionConfig': 'depth_video_diffusion',
    'DepthVideoDiffusion': 'depth_video_diffusion',
    'VGGTInterface': 'vggt_interface',
    'GaussianModel': 'gs_scene',
    'SceneRepresentation': 'gs_scene',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f'.{_EXPORTS[name]}', __name__), name)
    globals()[name] = value
    return value

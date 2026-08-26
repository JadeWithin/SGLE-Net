import torch

from rssgle.model import MSSSBackbone, RSSGLE, RSSGLEConfig


def test_backbone_dense_shapes() -> None:
    model = MSSSBackbone(bands=12, classes=4, widths=(16, 8), kernels=(3, 5, 7))
    cube = torch.randn(10, 9, 12)
    assert model.features(cube).shape == (10, 9, 8)
    assert model(cube).shape == (10, 9, 4)


def test_rssgle_forward_shape() -> None:
    model = RSSGLE(RSSGLEConfig(bands=10, classes=3, radius=1))
    cube = torch.randn(6, 5, 10)
    assert model(cube).shape == (6, 5, 3)


def test_houston_deployment_parameter_count() -> None:
    model = RSSGLE(RSSGLEConfig(bands=144, classes=15))
    assert sum(parameter.numel() for parameter in model.parameters()) == 170_622

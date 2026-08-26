import torch

from rssgle.sgle import SGLEConfig, sgle_refine


def test_sgle_shape_and_finiteness() -> None:
    torch.manual_seed(7)
    cube = torch.randn(9, 8, 6)
    logits = torch.randn(9, 8, 4)
    output = sgle_refine(cube, logits, SGLEConfig(radius=2))
    assert output.shape == logits.shape
    assert torch.isfinite(output).all()


def test_sgle_constant_logits_preserve_class_ties() -> None:
    cube = torch.randn(7, 6, 5)
    logits = torch.zeros(7, 6, 3)
    output = sgle_refine(cube, logits)
    assert torch.allclose(output[..., 0], output[..., 1])
    assert torch.allclose(output[..., 1], output[..., 2])

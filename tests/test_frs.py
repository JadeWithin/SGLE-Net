import torch

from rssgle.frs import NormalizedResidualHead, fold_frs_heads


def test_fold_matches_two_head_blend() -> None:
    torch.manual_seed(11)
    features = torch.randn(17, 8)
    mean_o, scale_o = torch.randn(8), torch.rand(8) + 0.3
    mean_q, scale_q = torch.randn(8), torch.rand(8) + 0.3
    ordinary = NormalizedResidualHead(mean_o, scale_o, 5)
    rotation = NormalizedResidualHead(mean_q, scale_q, 5)
    with torch.no_grad():
        ordinary.linear.weight.normal_()
        ordinary.linear.bias.normal_()
        rotation.linear.weight.normal_()
        rotation.linear.bias.normal_()
    alpha = 0.4
    expected = ordinary.residual(features) + alpha * (
        rotation.residual(features) - ordinary.residual(features)
    )
    folded = fold_frs_heads(ordinary, rotation, alpha)
    assert torch.allclose(folded(features), expected, atol=2e-6, rtol=1e-5)

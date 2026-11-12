import pytest
from crossfhe.pipeline import compile_torch, run_encrypted

pytestmark = pytest.mark.fhe


def test_grouped_dilated_conv_and_batchnorm_match_pytorch(tmp_path):
    import torch
    from torch import nn
    torch.manual_seed(19)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 2, groups=2, dilation=2, padding=1)
            self.bn = nn.BatchNorm2d(2)
        def forward(self, x): return self.bn(self.conv(x)).mean(dim=(2, 3))
    x = torch.linspace(-0.25, 0.25, 18).reshape(1, 2, 3, 3)
    artifact, arrays, reference = compile_torch(Model(), (x,), tmp_path/"conv", backend="seal-ckks")
    _, report = run_encrypted(artifact, arrays, reference=reference, atol=0.001, rtol=0.001)
    assert report["validation"]["passed"]


def test_batched_secret_matmul_and_broadcast_match_pytorch(tmp_path):
    import torch
    from torch import nn
    class Model(nn.Module):
        def forward(self, a, b): return (a@b.transpose(-1, -2)+0.25).sum(dim=-1, keepdim=True)
    a = torch.tensor([[[-0.5, 0.25, 0.75], [-0.25, 0.5, 0.125]]])
    b = torch.tensor([[[0.5, -0.25, 0.125], [0.25, 0.25, -0.5]]])
    artifact, arrays, reference = compile_torch(Model(), (a, b), tmp_path/"matmul", backend="seal-ckks")
    _, report = run_encrypted(artifact, arrays, reference=reference, atol=0.001, rtol=0.001)
    assert report["validation"]["passed"]


def test_constant_output_materialization_and_zero_multiply(tmp_path):
    import torch
    from torch import nn
    class Model(nn.Module):
        def forward(self, x): return x*0.0, torch.tensor([0.25, -0.125])
    artifact, arrays, reference = compile_torch(Model(), (torch.tensor([0.5, -0.25]),), tmp_path/"zeros", backend="seal-ckks")
    _, report = run_encrypted(artifact, arrays, reference=reference, atol=0.001, rtol=0.001)
    assert report["validation"]["passed"]

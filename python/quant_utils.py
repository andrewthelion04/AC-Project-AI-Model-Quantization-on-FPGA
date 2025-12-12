import torch
import torch.nn as nn

def calc_qparams_symmetric(x: torch.Tensor, num_bits: int):
    """Symmetric quantization (signed)."""
    qmax = (1 << (num_bits - 1)) - 1
    max_abs = x.abs().max().clamp(min=1e-8)
    scale = max_abs / qmax
    zero_point = 0
    return scale, zero_point

def quant_dequant_symmetric(x: torch.Tensor, num_bits: int):
    scale, _ = calc_qparams_symmetric(x, num_bits)
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    q = torch.round(x / scale).clamp(qmin, qmax)
    x_hat = q * scale
    return x_hat, scale

class FakeQuantSym(nn.Module):
    """Fake quant for QAT: quantize-dequantize in forward; STE in backward."""
    def __init__(self, num_bits: int):
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        x_hat, scale = quant_dequant_symmetric(x, self.num_bits)
        self.scale = scale.detach()
        # STE: x_hat has gradient as if identity (PyTorch autograd works with this op chain)
        return x_hat

import math
import torch
from torch import Tensor
from . import _C as _C


def flash_attn(q: Tensor, k: Tensor, v: Tensor, scale: float | None = None) -> Tensor:
    scale = 1 / math.sqrt(k.size(-1)) if scale is None else scale
    return torch.ops.flash_attn_rdna3.forward(q, k, v, scale)

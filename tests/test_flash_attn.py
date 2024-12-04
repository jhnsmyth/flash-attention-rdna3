from flash_attn_rdna3 import flash_attn
import torch
import torch.nn.functional as F
import math
import pytest


def attn(q, k, v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    o = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1) @ v
    return o.transpose(1, 2)


@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn(q_len, kv_len, head_dim, dtype):
    torch.manual_seed(0)
    batch_size = 4
    n_heads = 6
    device = "cuda"
    q = torch.randn(batch_size, q_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, kv_len, n_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    o = flash_attn(q, k, v)
    o_pt = attn(q, k, v)
    o_ref = attn(q.float(), k.float(), v.float()).to(dtype)
    assert (
        (o - o_ref).abs().max().item() <= 2 * (o_pt - o_ref).abs().max().item()
    ), "FlashAttention-RDNA3's numerical error should be at most twice the numerical error of a PyTorch implementation"

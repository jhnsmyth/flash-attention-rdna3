#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

namespace flash {

using bfloat = __bf16;
using bfloat2 = __bf16 __attribute__((ext_vector_type(2)));
using bfloat16 = __bf16 __attribute__((ext_vector_type(16)));

using float8 = float __attribute__((ext_vector_type(8)));

using half = __fp16;
using half2 = __fp16 __attribute__((ext_vector_type(2)));
using half16 = __fp16 __attribute__((ext_vector_type(16)));

template <typename T> struct trait;

template <> struct trait<bfloat> {
  using vector2 = bfloat2;
  using vector16 = bfloat16;
};

template <> struct trait<half> {
  using vector2 = half2;
  using vector16 = half16;
};

template <int pattern, typename T> __device__ T swizzle(T src) {
  return __builtin_bit_cast(
      T, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, src), pattern));
}

template <typename T>
__device__ float8 wmma_f32_16x16x16_w32(T a, T b, float8 c);

template <>
__device__ float8 wmma_f32_16x16x16_w32(bfloat16 a, bfloat16 b, float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

template <>
__device__ float8 wmma_f32_16x16x16_w32(half16 a, half16 b, float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

template <typename T, int D, int num_wavefronts>
__launch_bounds__(num_wavefronts * 32) __global__
    void kernel(T *o, const T *q, const T *k, const T *v, int64_t q_len,
                int64_t kv_len, float scale_log2_e) {
  static_assert(D % 16 == 0);
  static_assert(sizeof(T) == 2);
  static_assert(num_wavefronts * 32 * 8 == 16 * D);

  using vector2 = typename trait<T>::vector2;
  using vector16 = typename trait<T>::vector16;

  __shared__ T v_shared[16 * D];

  vector16 q_frag[D / 16];
  vector16 k_frag[D / 16];
  vector16 v_frag[D / 16];
  float8 o_frag[D / 16] = {};

  float l = 0;
  float mj_1 = -INFINITY;

  auto wid = threadIdx.x / 32;
  auto lid = threadIdx.x;
  auto lane = lid % 16;

  o = &o[q_len * gridDim.y * D * blockIdx.z];
  q = &q[q_len * gridDim.y * D * blockIdx.z];
  k = &k[kv_len * gridDim.y * D * blockIdx.z];
  v = &v[kv_len * gridDim.y * D * blockIdx.z];

  o = &o[D * (gridDim.y * (16 * (num_wavefronts * blockIdx.x + wid) + lane) +
              blockIdx.y)];
  q = &q[D * (gridDim.y * (16 * (num_wavefronts * blockIdx.x + wid) + lane) +
              blockIdx.y)];
  k = &k[D * (gridDim.y * lane + blockIdx.y)];
  v = &v[D * (gridDim.y * (lid / (D / 8)) + blockIdx.y) + 8 * (lid % (D / 8))];

  if (16 * num_wavefronts * blockIdx.x + 16 * wid + lane < q_len) {
    for (int d = 0; d < D / 16; ++d) {
      for (int ele = 0; ele < 16; ++ele) {
        q_frag[d][ele] = q[16 * d + ele];
      }
    }
  }

  auto j = 0u;
  for (; j < kv_len / 16; ++j) {
    for (auto d = 0u; d < D / 16; ++d) {
      for (auto ele = 0u; ele < 16; ++ele) {
        k_frag[d][ele] = k[16 * (D * gridDim.y * j + d) + ele];
      }
    }

    for (auto ele = 0u; ele < 8; ++ele) {
      v_shared[8 * lid + ele] = v[16 * D * gridDim.y * j + ele];
    }

    __syncthreads();

    for (auto d = 0u; d < D / 16; ++d) {
      for (auto ele = 0u; ele < 16; ++ele) {
        v_frag[d][ele] = v_shared[D * ele + 16 * d + lane];
      }
    }

    float8 s_frag = {};
    for (int d = 0; d < D / 16; ++d) {
      s_frag = wmma_f32_16x16x16_w32(k_frag[d], q_frag[d], s_frag);
    }

    float mj = mj_1;
    for (int ele = 0; ele < 8; ++ele) {
      mj = max(mj, s_frag[ele]);
    }
    mj = max(mj, __hip_ds_swizzlef(mj, 0x401f));

    for (int ele = 0; ele < 8; ++ele) {
      s_frag[ele] = exp2f((s_frag[ele] - mj) * scale_log2_e);
    }

    float scale = exp2f((mj_1 - mj) * scale_log2_e);

    l *= scale;
    for (int ele = 0; ele < 8; ++ele) {
      l += s_frag[ele];
    }

    vector2 tmp[4];
    for (int ele = 0; ele < 8; ++ele) {
      tmp[ele / 2][ele % 2] = static_cast<T>(s_frag[ele]);
    }

    vector16 p_frag;
    for (int ele = 0; ele < 4; ++ele) {
      vector2 tmp0 = swizzle<0xf>(tmp[ele]);
      vector2 tmp1 = swizzle<0x21f>(tmp[ele]);
      p_frag[ele * 4] = tmp0[0];
      p_frag[ele * 4 + 1] = tmp1[0];
      p_frag[ele * 4 + 2] = tmp0[1];
      p_frag[ele * 4 + 3] = tmp1[1];
    }

    for (int d = 0; d < D / 16; ++d) {
      for (int ele = 0; ele < 8; ++ele) {
        o_frag[d][ele] *= scale;
      }

      o_frag[d] = wmma_f32_16x16x16_w32(v_frag[d], p_frag, o_frag[d]);
    }

    mj_1 = mj;

    __syncthreads();
  }

  if (kv_len % 16 != 0) {
    if (lane < kv_len % 16) {
      for (auto d = 0u; d < D / 16; ++d) {
        for (auto ele = 0u; ele < 16; ++ele) {
          k_frag[d][ele] = k[16 * (D * gridDim.y * j + d) + ele];
        }
      }
    } else {
      for (int d = 0; d < D / 16; ++d) {
        for (int ele = 0; ele < 16; ++ele) {
          k_frag[d][ele] = 0;
        }
      }
    }

    if (lid / (D / 8) < kv_len % 16) {
      for (auto ele = 0u; ele < 8; ++ele) {
        v_shared[8 * lid + ele] = v[16 * D * gridDim.y * j + ele];
      }
    } else {
      for (auto ele = 0u; ele < 8; ++ele) {
        v_shared[8 * lid + ele] = 0;
      }
    }

    __syncthreads();

    for (auto d = 0u; d < D / 16; ++d) {
      for (auto ele = 0u; ele < 16; ++ele) {
        v_frag[d][ele] = v_shared[D * ele + 16 * d + lane];
      }
    }

    float8 s_frag = {};
    for (auto ele = 0u; ele < 8; ++ele) {
      if (ele * 2 + lid % 32 / 16 >= kv_len % 16) {
        s_frag[ele] = -INFINITY;
      }
    }

    for (int d = 0; d < D / 16; ++d) {
      s_frag = wmma_f32_16x16x16_w32(k_frag[d], q_frag[d], s_frag);
    }

    float mj = mj_1;
    for (int ele = 0; ele < 8; ++ele) {
      mj = max(mj, s_frag[ele]);
    }
    mj = max(mj, __hip_ds_swizzlef(mj, 0x401f));

    for (int ele = 0; ele < 8; ++ele) {
      s_frag[ele] = exp2f((s_frag[ele] - mj) * scale_log2_e);
    }

    float scale = exp2f((mj_1 - mj) * scale_log2_e);

    l *= scale;
    for (int ele = 0; ele < 8; ++ele) {
      l += s_frag[ele];
    }

    vector2 tmp[4];
    for (int ele = 0; ele < 8; ++ele) {
      tmp[ele / 2][ele % 2] = static_cast<T>(s_frag[ele]);
    }

    vector16 p_frag;
    for (int ele = 0; ele < 4; ++ele) {
      vector2 tmp0 = swizzle<0xf>(tmp[ele]);
      vector2 tmp1 = swizzle<0x21f>(tmp[ele]);
      p_frag[ele * 4] = tmp0[0];
      p_frag[ele * 4 + 1] = tmp1[0];
      p_frag[ele * 4 + 2] = tmp0[1];
      p_frag[ele * 4 + 3] = tmp1[1];
    }

    for (int d = 0; d < D / 16; ++d) {
      for (int ele = 0; ele < 8; ++ele) {
        o_frag[d][ele] *= scale;
      }

      o_frag[d] = wmma_f32_16x16x16_w32(v_frag[d], p_frag, o_frag[d]);
    }
  }

  if (16 * num_wavefronts * blockIdx.x + 16 * wid + lane >= q_len) {
    return;
  }

  l += __hip_ds_swizzlef(l, 0x401f);

  for (int d = 0; d < D / 16; ++d) {
    for (int ele = 0; ele < 8; ++ele) {
      o_frag[d][ele] /= l;
    }

    vector2 tmp[4];
    for (int ele = 0; ele < 8; ++ele) {
      tmp[ele / 2][ele % 2] = static_cast<T>(o_frag[d][ele]);
    }

    vector16 out;
    for (int ele = 0; ele < 4; ++ele) {
      vector2 tmp0 = swizzle<0xf>(tmp[ele]);
      vector2 tmp1 = swizzle<0x21f>(tmp[ele]);
      out[ele * 4] = tmp0[0];
      out[ele * 4 + 1] = tmp1[0];
      out[ele * 4 + 2] = tmp0[1];
      out[ele * 4 + 3] = tmp1[1];
    }

    for (int ele = 0; ele < 8; ++ele) {
      const int col = lid % 32 / 16;
      o[16 * d + col * 8 + ele] = out[col * 8 + ele];
    }
  }
}

constexpr float LOG2_E = 0x1.715476p+0f;

template <typename T, typename U> T narrow(U value) {
  auto result = static_cast<T>(value);
  TORCH_CHECK(static_cast<U>(result) == value);
  return result;
}

at::Tensor forward(const at::Tensor &q, const at::Tensor &k,
                   const at::Tensor &v, const double scale) {
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(k.is_contiguous());
  TORCH_CHECK(v.is_contiguous());
  TORCH_CHECK(q.dim() == 4);
  TORCH_CHECK(k.dim() == 4);
  TORCH_CHECK(v.dim() == 4);

  auto scalar_type = q.scalar_type();
  TORCH_CHECK(scalar_type == k.scalar_type());
  TORCH_CHECK(scalar_type == v.scalar_type());

  auto batch_size = narrow<uint32_t>(q.size(0));
  TORCH_CHECK(batch_size == k.size(0));
  TORCH_CHECK(batch_size == v.size(0));

  auto q_len = q.size(1);

  auto kv_len = k.size(1);
  TORCH_CHECK(kv_len == v.size(1));

  auto num_heads = narrow<uint32_t>(q.size(2));
  TORCH_CHECK(num_heads == k.size(2));
  TORCH_CHECK(num_heads == v.size(2));

  auto head_dim = q.size(3);
  TORCH_CHECK(head_dim == k.size(3));
  TORCH_CHECK(head_dim == v.size(3));

  auto device_type = q.device().type();
  TORCH_CHECK(device_type == k.device().type());
  TORCH_CHECK(device_type == v.device().type());
  TORCH_CHECK(device_type == at::DeviceType::CUDA);

  auto device_index = q.device().index();
  TORCH_CHECK(device_index == k.device().index());
  TORCH_CHECK(device_index == v.device().index());

  hipDeviceProp_t prop;
  TORCH_CHECK(hipGetDeviceProperties(&prop, device_index) == hipSuccess);

  bool is_gfx11 = strcmp(prop.gcnArchName, "gfx1100") == 0 ||
                  strcmp(prop.gcnArchName, "gfx1101") == 0;
  TORCH_CHECK(is_gfx11, "FlashAttention-RDNA3 only supports RDNA3 GPUs");

  auto o = at::empty_like(q);

  if (q_len == 0 || num_heads == 0 || batch_size == 0) {
    return o;
  }

  if (kv_len == 0) {
    o.zero_();
    return o;
  }

  auto scale_log2_e = static_cast<float>(scale * LOG2_E);

  auto num_blocks = narrow<uint32_t>((q_len + 63) / 64);

  auto launch_kernel = [&]<typename T, int D, int num_wavefronts>() {
    TORCH_CHECK(q_len * num_heads * D <= UINT_MAX);
    TORCH_CHECK(kv_len * num_heads * D <= UINT_MAX);

    kernel<T, D, num_wavefronts>
        <<<dim3(num_blocks, num_heads, batch_size), dim3(num_wavefronts * 32),
           0, at::hip::getCurrentHIPStream(device_index)>>>(
            static_cast<T *>(o.data_ptr()), static_cast<T *>(q.data_ptr()),
            static_cast<T *>(k.data_ptr()), static_cast<T *>(v.data_ptr()),
            q_len, kv_len, scale_log2_e);
  };

  auto launch_kernel_with_type = [&]<typename T>() {
    switch (head_dim) {
    case 64:
      launch_kernel.template operator()<T, 64, 4>();
      break;
    default:
      TORCH_CHECK(
          false,
          "FlashAttention-RDNA3 only supports head dimension 64 for now");
      break;
    }
  };

  switch (scalar_type) {
  case at::kBFloat16:
    launch_kernel_with_type.template operator()<bfloat>();
    break;
  case at::kHalf:
    launch_kernel_with_type.template operator()<half>();
    break;
  default:
    TORCH_CHECK(false,
                "FlashAttention-RDNA3 only supports bfloat16 or float16");
    break;
  }

  return o;
}

PYBIND11_MODULE(_C, m) { m.doc() = "FlashAttention-RDNA3"; }

TORCH_LIBRARY(flash_attn_rdna3, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_attn_rdna3, CUDA, m) { m.impl("forward", &forward); }

} // namespace flash

// Copyright 2026 Seedleap.ai
// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace {

constexpr uint32_t kZingHeadDim = 128;
constexpr uint32_t kZingHalfHeadDim = kZingHeadDim / 2;
constexpr uint32_t kZingThreadsPerToken = 16;
constexpr uint32_t kZingThreadsPerBlock = 256;
constexpr uint32_t kZingTokensPerBlock = kZingThreadsPerBlock / kZingThreadsPerToken;
constexpr uint32_t kZingElementsPerLane = kZingHeadDim / kZingThreadsPerToken;
constexpr uint32_t kZingPairsPerLane = kZingElementsPerLane / 2;

static_assert(kZingElementsPerLane * sizeof(bf16_t) == device::kMaxVecBytes);
static_assert(kZingPairsPerLane * sizeof(float) == device::kMaxVecBytes);

struct ZingRotaryParams {
  const void* __restrict__ input;
  const float* __restrict__ cos;
  const float* __restrict__ sin;
  void* __restrict__ output;
  uint32_t num_tokens;
  uint32_t sequence_length;
  uint32_t num_heads;
};

template <bool kUsePDL, typename DType>
__global__
__launch_bounds__(kZingThreadsPerBlock) void zing_rotary_sm90(const ZingRotaryParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  using HiddenVector = AlignedVector<DType, kZingElementsPerLane>;
  using TableVector = AlignedVector<float, kZingPairsPerLane>;

  const uint32_t token_in_block = threadIdx.x / kZingThreadsPerToken;
  const uint32_t lane_in_token = threadIdx.x % kZingThreadsPerToken;
  const uint32_t num_workers = gridDim.x * kZingTokensPerBlock;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t token = blockIdx.x * kZingTokensPerBlock + token_in_block; token < params.num_tokens;
       token += num_workers) {
    const uint32_t sequence_idx = token % params.sequence_length;
    const auto input =
        static_cast<const DType*>(params.input) + static_cast<int64_t>(token) * params.num_heads * kZingHeadDim;
    auto output = static_cast<DType*>(params.output) + static_cast<int64_t>(token) * params.num_heads * kZingHeadDim;

    TableVector cos;
    TableVector sin;
    cos.load(params.cos + static_cast<int64_t>(sequence_idx) * kZingHalfHeadDim, lane_in_token);
    sin.load(params.sin + static_cast<int64_t>(sequence_idx) * kZingHalfHeadDim, lane_in_token);

#pragma unroll 1
    for (uint32_t head = 0; head < params.num_heads; ++head) {
      HiddenVector x;
      HiddenVector y;
      x.load(input + static_cast<int64_t>(head) * kZingHeadDim, lane_in_token);

#pragma unroll
      for (uint32_t pair = 0; pair < kZingPairsPerLane; ++pair) {
        const float real = cast<float>(x[2 * pair]);
        const float imaginary = cast<float>(x[2 * pair + 1]);
        const float real_cos = __fmul_rn(real, cos[pair]);
        const float imaginary_sin = __fmul_rn(imaginary, sin[pair]);
        const float real_sin = __fmul_rn(real, sin[pair]);
        const float imaginary_cos = __fmul_rn(imaginary, cos[pair]);
        y[2 * pair] = cast<DType>(__fsub_rn(real_cos, imaginary_sin));
        y[2 * pair + 1] = cast<DType>(__fadd_rn(real_sin, imaginary_cos));
      }
      y.store(output + static_cast<int64_t>(head) * kZingHeadDim, lane_in_token);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, typename DType>
struct ZingRotaryKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView cos,
      const tvm::ffi::TensorView sin,
      const tvm::ffi::TensorView output,
      int64_t sequence_length) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto S = SymbolicSize{"sequence_length"};
    auto R = SymbolicSize{"half_head_dim"};
    auto device = SymbolicDevice{};
    D.set_value(kZingHeadDim);
    R.set_value(kZingHalfHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D})
        .with_strides({-1, kZingHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input)
        .verify(output);
    TensorMatcher({S, R})
        .with_strides({kZingHalfHeadDim, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(cos)
        .verify(sin);

    RuntimeCheck(sequence_length == S.unwrap(), "zing_rotary: sequence length mismatch");
    RuntimeCheck(sequence_length > 0, "zing_rotary: sequence length must be positive");
    RuntimeCheck(N.unwrap() % sequence_length == 0, "zing_rotary: token count must contain whole sequences");

    const auto params = ZingRotaryParams{
        .input = input.data_ptr(),
        .cos = static_cast<const float*>(cos.data_ptr()),
        .sin = static_cast<const float*>(sin.data_ptr()),
        .output = output.data_ptr(),
        .num_tokens = static_cast<uint32_t>(N.unwrap()),
        .sequence_length = static_cast<uint32_t>(sequence_length),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };

    constexpr auto kernel = zing_rotary_sm90<kUsePDL, DType>;
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kZingThreadsPerBlock);
    const uint32_t num_sms = runtime::get_sm_count(device.unwrap().device_id);
    const uint32_t needed_blocks = div_ceil(params.num_tokens, kZingTokensPerBlock);
    const uint32_t num_blocks = std::min(needed_blocks, max_occupancy * num_sms);
    LaunchKernel(num_blocks, kZingThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace

}  // namespace sglang

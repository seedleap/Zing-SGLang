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

namespace {

constexpr uint32_t kMinWMHeadDim = 128;
constexpr uint32_t kMinWMHalfHeadDim = kMinWMHeadDim / 2;
constexpr uint32_t kMinWMThreadsPerToken = 16;
constexpr uint32_t kMinWMThreadsPerBlock = 256;
constexpr uint32_t kMinWMTokensPerBlock = kMinWMThreadsPerBlock / kMinWMThreadsPerToken;
constexpr uint32_t kMinWMElementsPerLane = kMinWMHeadDim / kMinWMThreadsPerToken;
constexpr uint32_t kMinWMPairsPerLane = kMinWMElementsPerLane / 2;

static_assert(kMinWMElementsPerLane * sizeof(bf16_t) == device::kMaxVecBytes);
static_assert(kMinWMPairsPerLane * sizeof(float) == device::kMaxVecBytes);

struct MinWMRotaryParams {
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
__launch_bounds__(kMinWMThreadsPerBlock) void minwm_rotary_sm90(const MinWMRotaryParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  using HiddenVector = AlignedVector<DType, kMinWMElementsPerLane>;
  using TableVector = AlignedVector<float, kMinWMPairsPerLane>;

  const uint32_t token_in_block = threadIdx.x / kMinWMThreadsPerToken;
  const uint32_t lane_in_token = threadIdx.x % kMinWMThreadsPerToken;
  const uint32_t num_workers = gridDim.x * kMinWMTokensPerBlock;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t token = blockIdx.x * kMinWMTokensPerBlock + token_in_block; token < params.num_tokens;
       token += num_workers) {
    const uint32_t sequence_idx = token % params.sequence_length;
    const auto input =
        static_cast<const DType*>(params.input) + static_cast<int64_t>(token) * params.num_heads * kMinWMHeadDim;
    auto output = static_cast<DType*>(params.output) + static_cast<int64_t>(token) * params.num_heads * kMinWMHeadDim;

    TableVector cos;
    TableVector sin;
    cos.load(params.cos + static_cast<int64_t>(sequence_idx) * kMinWMHalfHeadDim, lane_in_token);
    sin.load(params.sin + static_cast<int64_t>(sequence_idx) * kMinWMHalfHeadDim, lane_in_token);

#pragma unroll 1
    for (uint32_t head = 0; head < params.num_heads; ++head) {
      HiddenVector x;
      HiddenVector y;
      x.load(input + static_cast<int64_t>(head) * kMinWMHeadDim, lane_in_token);

#pragma unroll
      for (uint32_t pair = 0; pair < kMinWMPairsPerLane; ++pair) {
        const float real = cast<float>(x[2 * pair]);
        const float imaginary = cast<float>(x[2 * pair + 1]);
        const float real_cos = __fmul_rn(real, cos[pair]);
        const float imaginary_sin = __fmul_rn(imaginary, sin[pair]);
        const float real_sin = __fmul_rn(real, sin[pair]);
        const float imaginary_cos = __fmul_rn(imaginary, cos[pair]);
        y[2 * pair] = cast<DType>(__fsub_rn(real_cos, imaginary_sin));
        y[2 * pair + 1] = cast<DType>(__fadd_rn(real_sin, imaginary_cos));
      }
      y.store(output + static_cast<int64_t>(head) * kMinWMHeadDim, lane_in_token);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, typename DType>
struct MinWMRotaryKernel {
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
    D.set_value(kMinWMHeadDim);
    R.set_value(kMinWMHalfHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D})
        .with_strides({-1, kMinWMHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input)
        .verify(output);
    TensorMatcher({S, R})
        .with_strides({kMinWMHalfHeadDim, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(cos)
        .verify(sin);

    RuntimeCheck(sequence_length == S.unwrap(), "minwm_rotary: sequence length mismatch");
    RuntimeCheck(sequence_length > 0, "minwm_rotary: sequence length must be positive");
    RuntimeCheck(N.unwrap() % sequence_length == 0, "minwm_rotary: token count must contain whole sequences");

    const auto params = MinWMRotaryParams{
        .input = input.data_ptr(),
        .cos = static_cast<const float*>(cos.data_ptr()),
        .sin = static_cast<const float*>(sin.data_ptr()),
        .output = output.data_ptr(),
        .num_tokens = static_cast<uint32_t>(N.unwrap()),
        .sequence_length = static_cast<uint32_t>(sequence_length),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };

    constexpr auto kernel = minwm_rotary_sm90<kUsePDL, DType>;
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kMinWMThreadsPerBlock);
    const uint32_t num_sms = runtime::get_sm_count(device.unwrap().device_id);
    const uint32_t needed_blocks = div_ceil(params.num_tokens, kMinWMTokensPerBlock);
    const uint32_t num_blocks = std::min(needed_blocks, max_occupancy * num_sms);
    LaunchKernel(num_blocks, kMinWMThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace

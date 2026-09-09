// Copyright 2026 Seedleap.ai
// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

struct MinWMRMSNormParams {
  const void* q_input;
  const void* k_input;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  void* q_output;
  void* k_output;
  int64_t q_input_stride;
  int64_t k_input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  uint32_t num_inputs;
  float eps;
};

// One full-width MinWM row is 6 KiB in fp16/bf16. Assign one naturally aligned
// maximum-width vector to each thread: 384 x 16B on Hopper and 192 x 32B on
// Blackwell. A CTA therefore consumes a row in one coalesced pass while keeping
// the input resident in registers across the reduction.
template <int64_t kDim, typename DType>
inline constexpr uint32_t kMinWMThreadsPerBlock = kDim / (device::kMaxVecBytes / sizeof(DType));

template <int64_t kDim, bool kUsePDL, typename DType>
__global__ __launch_bounds__(kMinWMThreadsPerBlock<kDim, DType>) void minwm_rmsnorm_vector_cta(
    const MinWMRMSNormParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  using Packed = packed_t<DType>;
  constexpr uint32_t kPackedPerVector = kMaxVecBytes / sizeof(Packed);
  constexpr uint32_t kElementsPerVector = kMaxVecBytes / sizeof(DType);
  constexpr uint32_t kThreadsPerBlock = kMinWMThreadsPerBlock<kDim, DType>;
  constexpr uint32_t kNumWarps = kThreadsPerBlock / kWarpThreads;
  using Storage = AlignedVector<Packed, kPackedPerVector>;

  static_assert(kDim == 3072, "MinWM vector CTA is specialized for hidden_size=3072");
  static_assert(kDim % kElementsPerVector == 0);
  static_assert(kThreadsPerBlock % kWarpThreads == 0);
  static_assert(kNumWarps <= kWarpThreads);

  const auto& [q_input, k_input, q_weight, k_weight, q_output, k_output, q_input_stride, k_input_stride, output_stride, num_tokens, num_inputs, eps] =
      params;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t num_works = num_tokens * num_inputs;
  __shared__ float reduction[kNumWarps + 1];

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t work_id = blockIdx.x; work_id < num_works; work_id += gridDim.x) {
    const bool is_k = work_id >= num_tokens;
    const uint32_t token_id = is_k ? work_id - num_tokens : work_id;
    const auto input =
        pointer::offset<DType>(is_k ? k_input : q_input, token_id * (is_k ? k_input_stride : q_input_stride));
    const auto weight = is_k ? k_weight : q_weight;
    const auto output = pointer::offset<DType>(is_k ? k_output : q_output, token_id * output_stride);

    Storage input_cache;
    input_cache.load(input, threadIdx.x);
    float sum_of_squares = 0.0f;
#pragma unroll
    for (uint32_t packed_idx = 0; packed_idx < kPackedPerVector; ++packed_idx) {
      const auto [x0, x1] = cast<fp32x2_t>(input_cache[packed_idx]);
      sum_of_squares += x0 * x0 + x1 * x1;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) {
      reduction[warp_id] = sum_of_squares;
    }
    __syncthreads();
    if (warp_id == 0) {
      float warp_sum = lane_id < kNumWarps ? reduction[lane_id] : 0.0f;
      warp_sum = warp::reduce_sum(warp_sum);
      if (lane_id == 0) {
        reduction[kNumWarps] = math::rsqrt(warp_sum / static_cast<float>(kDim) + eps);
      }
    }
    __syncthreads();
    const float norm_factor = reduction[kNumWarps];

    Storage weight_vec;
    Storage output_vec;
    weight_vec.load(weight, threadIdx.x);
#pragma unroll
    for (uint32_t packed_idx = 0; packed_idx < kPackedPerVector; ++packed_idx) {
      const auto [x0, x1] = cast<fp32x2_t>(input_cache[packed_idx]);
      const Packed normalized = cast<Packed>(fp32x2_t{x0 * norm_factor, x1 * norm_factor});
      const auto [n0, n1] = cast<fp32x2_t>(normalized);
      const auto [w0, w1] = cast<fp32x2_t>(weight_vec[packed_idx]);
      output_vec[packed_idx] = cast<Packed>(fp32x2_t{n0 * w0, n1 * w1});
    }
    output_vec.store(output, threadIdx.x);
    __syncthreads();
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename DType>
void launch_minwm_rmsnorm(const MinWMRMSNormParams& params, const DLDevice device) {
  constexpr auto kernel = minwm_rmsnorm_vector_cta<kDim, kUsePDL, DType>;
  constexpr uint32_t threads_per_block = kMinWMThreadsPerBlock<kDim, DType>;
  static const uint32_t max_occupancy = host::runtime::get_blocks_per_sm(kernel, threads_per_block);
  static const uint32_t num_sms = host::runtime::get_sm_count(device.device_id);
  const uint32_t num_works = params.num_tokens * params.num_inputs;
  const uint32_t num_blocks = std::min(num_works, max_occupancy * num_sms);
  host::LaunchKernel(num_blocks, threads_per_block, device).enable_pdl(kUsePDL)(kernel, params);
}

template <int64_t kDim, bool kUsePDL, typename DType>
struct MinWMRMSNormKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_strides({D, 1}).with_dtype<DType>().with_device(device).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "minwm_rmsnorm: num_tokens must be > 0");
    const auto params = MinWMRMSNormParams{
        .q_input = input.data_ptr(),
        .k_input = nullptr,
        .q_weight = weight.data_ptr(),
        .k_weight = nullptr,
        .q_output = output.data_ptr(),
        .k_output = nullptr,
        .q_input_stride = SI.unwrap(),
        .k_input_stride = 0,
        .output_stride = kDim,
        .num_tokens = num_tokens,
        .num_inputs = 1,
        .eps = eps,
    };
    launch_minwm_rmsnorm<kDim, kUsePDL, DType>(params, device.unwrap());
  }
};

template <int64_t kDim, bool kUsePDL, typename DType>
struct MinWMFusedQKNormKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView q_output,
      const tvm::ffi::TensorView k_output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SQ = SymbolicSize{"q_stride"};
    auto SK = SymbolicSize{"k_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SQ, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, D}).with_strides({SK, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({N, D}).with_strides({D, 1}).with_dtype<DType>().with_device(device).verify(q_output).verify(
        k_output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "minwm_fused_qknorm: num_tokens must be > 0");
    const auto params = MinWMRMSNormParams{
        .q_input = q.data_ptr(),
        .k_input = k.data_ptr(),
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .q_output = q_output.data_ptr(),
        .k_output = k_output.data_ptr(),
        .q_input_stride = SQ.unwrap(),
        .k_input_stride = SK.unwrap(),
        .output_stride = kDim,
        .num_tokens = num_tokens,
        .num_inputs = 2,
        .eps = eps,
    };
    launch_minwm_rmsnorm<kDim, kUsePDL, DType>(params, device.unwrap());
  }
};

}  // namespace

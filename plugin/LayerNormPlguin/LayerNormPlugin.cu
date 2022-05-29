#include <cuda.h>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "LayerNormPlugin.h"

#define FINAL_MASK 0xffffffff

using namespace nvinfer1;
using nvinfer1::plugin::LayerNormPlugin;
using nvinfer1::plugin::LayerNormPluginCreator;
using half = __half;

constexpr int kWarpSize             = 32;
constexpr float kR256               = 0.00390625f;
static constexpr float HALF_FLT_MAX = 65504.f;

__device__ __forceinline__ unsigned int LaneId() {
  unsigned int lane;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}

template <typename T>
__device__ __forceinline__ T ClampInfForHalf(const float val) {
  if (std::is_same<T, half>::value == true) {
    return (float)val > 0.0f ? min(val, HALF_FLT_MAX - 1000) : max(val, -HALF_FLT_MAX + 1000);
  } else {
    return val;
  }
}

template <typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
#pragma unroll
  for (int stride = kWarpSize / 2; stride >= 1; stride >>= 1) {
    val += __shfl_xor_sync(-1, val, stride);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T BlockReduceSum(T& val) {
  static __shared__ T shared[kWarpSize];
  unsigned int lane = threadIdx.x & 0x1f;
  unsigned int wid  = threadIdx.x >> 5;

  val = WarpReduceSum<T>(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : T(0.0f);
  val = WarpReduceSum<T>(val);
  return val;
}

template <typename T>
__device__ __forceinline__ void BlockDoubleReduceSum(T& val0, T& val1) {
  static __shared__ T shared0[kWarpSize];
  static __shared__ T shared1[kWarpSize];
  unsigned int lane = threadIdx.x & 0x1f;
  unsigned int wid  = threadIdx.x >> 5;

  val0 = WarpReduceSum<T>(val0);
  val1 = WarpReduceSum<T>(val1);

  if (lane == 0) {
    shared0[wid] = val0;
    shared1[wid] = val1;
  }
  __syncthreads();

  val0 = (lane < (blockDim.x >> 5)) ? shared0[lane] : T(0.0f);
  val1 = (lane < (blockDim.x >> 5)) ? shared1[lane] : T(0.0f);

  val0 = WarpReduceSum<T>(val0);
  val1 = WarpReduceSum<T>(val1);
  return;
}

template <typename T, typename P>
__global__ void LayerNormKernel(const T* __restrict input,
                                const P* __restrict gamma,
                                const P* __restrict beta,
                                T* __restrict output,
                                const int norm_size,
                                const float eps) {

  float2 temp;
  temp.x = 0.f;
  temp.y = 0.f;

  for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float val = input[blockIdx.x * norm_size + i];
    temp.x += val;
    temp.y += val * val;
  }

  BlockDoubleReduceSum(temp.x, temp.y);

  float mean = temp.x / norm_size;
  float rstd = rsqrtf(temp.y / norm_size - mean * mean + eps);

#pragma unroll
  for (int i = threadIdx.x; i < norm_size; i += blockDim.x) {
    const int index = blockIdx.x * norm_size + i;
    output[index] = ClampInfForHalf<T>((((float)input[index] - mean) * rstd * gamma[i] + beta[i]));
  }
}

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc,
                             const PluginTensorDesc* outputDesc,
                             const void* const* inputs,
                             void* const* outputs,
                             void* workspace,
                             cudaStream_t stream) noexcept {
  DEBUG_FUNC();
  const int input_volume = volume(inputDesc[0].dims);
  int status             = -1;
  DataType iType         = inputDesc->type;

  int norm_size = volume(inputDesc[1].dims);
  assert(input_volume % norm_size == 0);
  int num_instance = static_cast<int>(input_volume / norm_size);

  dim3 gridSize(num_instance);
  dim3 blockSize(norm_size);
  if (iType == DataType::kFLOAT) {
    const auto* const input = static_cast<const float*>(inputs[0]);
    const auto* const gamma = static_cast<const float*>(inputs[1]);
    const auto* const beta  = static_cast<const float*>(inputs[2]);
    auto* output            = static_cast<float*>(outputs[0]);
    LayerNormKernel<float, float>
        <<<gridSize, blockSize, 0, stream>>>(input, gamma, beta, output, norm_size, 1e-5f);
  } else if (iType == DataType::kHALF) {
    const auto* const input = static_cast<const half*>(inputs[0]);
    const auto* const gamma = static_cast<const float*>(inputs[1]);
    const auto* const beta  = static_cast<const float*>(inputs[2]);
    auto* output            = static_cast<half*>(outputs[0]);
    LayerNormKernel<half, float>
        <<<gridSize, blockSize, 0, stream>>>(input, gamma, beta, output, norm_size, 1e-5f);
  }
  return 0;
}

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attrs_{};
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
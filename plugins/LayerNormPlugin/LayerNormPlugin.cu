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
static constexpr float HALF_FLT_MAX = 65504.f;

template <typename T, typename OP_T, int TPB>
__global__ void LayerNormKernelSmall(const int norm_size,
                                     const T* input,
                                     const T* gamma,
                                     const T* beta,
                                     T* output) {
  const int index     = blockIdx.x * norm_size + threadIdx.x;
  const T denominator = T(1) / T(norm_size);
  OP_T val            = 0;
  kvp<OP_T> threadData(0, 0);

  if (threadIdx.x < norm_size) {
    val       = input[index] * denominator;
    OP_T tmp0 = val * (OP_T)denominator, tmp1 = val * tmp0;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
  }

  using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>()); // cub::Sum() 用不了？

  if (threadIdx.x == 0) {
    mu     = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)1e-5);
  }
  __syncthreads();

  if (threadIdx.x < norm_size) {
    const OP_T g = gamma[threadIdx.x], b = beta[threadIdx.x];
    output[index] = (val - mu) * rsigma * g + b;
  }
}

template __global__ void
LayerNormKernelSmall<float, float, 32>(const int, const float*, const float*, const float*, float*);
template __global__ void LayerNormKernelSmall<__half, float, 32>(const int,
                                                                 const __half*,
                                                                 const __half*,
                                                                 const __half*,
                                                                 __half*);

template <typename T, typename OP_T, int TPB, int VPT>
__global__ void LayerNormKernelMedium(const int norm_size,
                                      const T* input,
                                      const T* gamma,
                                      const T* beta,
                                      T* output) {
  // 考虑一个 block 上的寄存器使用量，当 norm_size 最大为 1024，元素为 float 时，
  // localX:      256 thread/block * 4 element/thread（即VPT） * 4 Byte/element = 4 KiB
  // localBeta:   1024 element / block * 4 Byte / element = 4 KiB
  // localGamma:  1024 element / block * 4 Byte / element = 4 KiB
  // localBias:   1024 element / block * 4 Byte / element = 4 KiB（这里没有）

  const int index = blockIdx.x * norm_size + threadIdx.x * VPT;
  T localX[VPT], localGamma[VPT], localBeta[VPT];
  const OP_T denominator = OP_T(1) / OP_T(norm_size);
  kvp<OP_T> threadData(0, 0);

  copy<sizeof(T) * VPT>(&input[index], localX);
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const OP_T tmp = (OP_T)localX[it] * denominator;
    threadData     = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * (OP_T)localX[it]));
  }

  copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu     = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)1e-5);
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    localX[it] = (OP_T)localGamma[it] * ((OP_T)localX[it] - mu) * rsigma + (OP_T)localBeta[it];
  }

  copy<sizeof(T) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormKernelMedium<float, float, 64, 4>(const int,
                                                                    const float*,
                                                                    const float*,
                                                                    const float*,
                                                                    float*);
template __global__ void LayerNormKernelMedium<__half, float, 64, 4>(const int,
                                                                     const __half*,
                                                                     const __half*,
                                                                     const __half*,
                                                                     __half*);

template <typename T, typename OP_T, int TPB>
__global__ void LayerNormKernelLarge(const int norm_size,
                                     const T* input,
                                     const T* gamma,
                                     const T* beta,
                                     T* output) {
  const int offset       = blockIdx.x * norm_size;
  const OP_T denominator = OP_T(1) / OP_T(norm_size);
  kvp<OP_T> threadData(0, 0);

  for (int i = threadIdx.x; i < norm_size; i += TPB) {
    const int index = offset + i;
    OP_T val        = input[index];
    const OP_T tmp  = val * denominator;
    threadData      = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
    output[index]   = val;
  }

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu     = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)1e-5);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < norm_size; i += TPB) {
    const int index = offset + i;
    output[index]   = ((OP_T)output[index] - mu) * rsigma * (OP_T)gamma[i] + (OP_T)beta[i];
  }
}

template __global__ void LayerNormKernelLarge<float, float, 256>(const int,
                                                                 const float*,
                                                                 const float*,
                                                                 const float*,
                                                                 float*);
template __global__ void LayerNormKernelLarge<__half, float, 256>(const int,
                                                                  const __half*,
                                                                  const __half*,
                                                                  const __half*,
                                                                  __half*);

template <int TPB, int VPT>
__global__ void LayerNormKernelQDQ(const int norm_size,
                                   const int8_t* input,
                                   int8_t* output,
                                   const __half* gamma,
                                   const __half* beta,
                                   const float dqScaleIn,
                                   const float qScale) {
  const int index = norm_size * blockIdx.x + threadIdx.x * VPT;
  int8_t localX[VPT];
  __half localXDQ[VPT], localBeta[VPT], localGamma[VPT];

  copy<sizeof(int8_t) * VPT>(&input[index], localX);
  __half2 loc = __floats2half2_rn(0.f, 0.f);

  const __half denominator = __half(1) / __half(norm_size);
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const float tmp_in = localX[it];
    localXDQ[it]       = dqScaleIn * tmp_in;

    const __half tmp   = localXDQ[it] * denominator;
    const __half2 tmp2 = __halves2half2(tmp, tmp * localXDQ[it]);
    loc                = loc + tmp2;
  }

  copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<__half2, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ __half mu;     // mean
  __shared__ __half rsigma; // 1 / std.dev.

  // const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());
  const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

  if (threadIdx.x == 0) {
    mu     = __low2half(sum2);
    rsigma = rsqrt(__high2half(sum2) - mu * mu);
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const float tmp = localGamma[it] * (localXDQ[it] - mu) * rsigma + localBeta[it];
    int tmpq        = __float2int_rn(qScale * tmp);
    tmpq            = max(-127, tmpq);
    tmpq            = min(127, tmpq);
    localX[it]      = tmpq;
  }

  copy<sizeof(int8_t) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormKernelQDQ<32, 8>(const int,
                                                   const int8_t* input,
                                                   int8_t* output,
                                                   const __half* gamma,
                                                   const __half* beta,
                                                   const float dqScaleIn,
                                                   const float qScale);
template __global__ void LayerNormKernelQDQ<128, 8>(const int,
                                                    const int8_t* input,
                                                    int8_t* output,
                                                    const __half* gamma,
                                                    const __half* beta,
                                                    const float dqScaleIn,
                                                    const float qScale);

template <typename T>
int DispatchLayerNorm(const int gridSize,
                     const int norm_size,
                     const T* input,
                     const T* gamma,
                     const T* beta,
                     T* output,
                     cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
  if (norm_size <= 32) {
    constexpr int TPB = 32;
    (LayerNormKernelSmall<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, gamma, beta, output);
  } else if (norm_size == 256) {
    constexpr int TPB = 256 / VPT;
    (LayerNormKernelMedium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, gamma, beta, output);
  } else if (norm_size == 1024) {
    constexpr int TPB = 1024 / VPT;
    (LayerNormKernelMedium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, gamma, beta, output);
  } else {
    constexpr int TPB = 256;
    (LayerNormKernelLarge<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, gamma, beta, output);
  }
  CHECK(cudaPeekAtLastError());
  return 0;
}

template int DispatchLayerNorm<float>(const int,
                                     const int,
                                     const float*,
                                     const float*,
                                     const float*,
                                     float*,
                                     cudaStream_t);
template int DispatchLayerNorm<half>(const int,
                                    const int,
                                    const half*,
                                    const half*,
                                    const half*,
                                    half*,
                                    cudaStream_t);

int DispatchLayerNormDQQ(const int gridSize,
                        const int norm_size,
                        const int8_t* input,
                        const __half* gamma,
                        const __half* beta,
                        int8_t* output,
                        const float dqScaleIn,
                        const float qScale,
                        cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(__half);
  if (norm_size == 256) {
    constexpr int TPB = 256 / VPT;
    (LayerNormKernelQDQ<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, output, gamma, beta, dqScaleIn, qScale);
  } else if (norm_size == 1024) {
    constexpr int TPB = 1024 / VPT;
    (LayerNormKernelQDQ<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        norm_size, input, output, gamma, beta, dqScaleIn, qScale);
  } else {
    printf("[DispatchLayerNormDQQ] Unsupport hidden dimension %d!\n", norm_size);
    exit(0);
  }
  CHECK(cudaPeekAtLastError());
  return 0;
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                 const PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs,
                                 void* workspace,
                                 cudaStream_t stream) noexcept {
  const int norm_size = volume(inputDesc[1].dims);
  const int grid      = volume(inputDesc[0].dims) / norm_size;
  int status          = -1;

  switch (int(inputDesc[0].type)) {
  case int(DataType::kFLOAT): {

    const auto input = static_cast<const float*>(inputs[0]);
    const auto gamma = static_cast<const float*>(inputs[1]);
    const auto beta  = static_cast<const float*>(inputs[2]);
    auto output      = static_cast<float*>(outputs[0]);

    status = DispatchLayerNorm<float>(grid, norm_size, input, gamma, beta, output, stream);
    break;
  }
  case int(DataType::kHALF): {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto gamma = static_cast<const half*>(inputs[1]);
    const auto beta  = static_cast<const half*>(inputs[2]);
    auto output      = static_cast<half*>(outputs[0]);

    status = DispatchLayerNorm<half>(grid, norm_size, input, gamma, beta, output, stream);
    break;
  }
  case int(DataType::kINT8): {
    const float dqScaleIn = inputDesc[0].scale;
    const float qScale    = 1.f / outputDesc[0].scale;
    const auto input      = static_cast<const int8_t*>(inputs[0]);
    auto output           = static_cast<int8_t*>(outputs[0]);
    const auto gamma      = static_cast<const half*>(inputs[1]);
    const auto beta       = static_cast<const half*>(inputs[2]);

    status =
        DispatchLayerNormDQQ(grid, norm_size, input, gamma, beta, output, dqScaleIn, qScale, stream);
    break;
  }
  default: {
    printf("DataType not support!\n");
  }
  }
  return status;
}
PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attrs_{};
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
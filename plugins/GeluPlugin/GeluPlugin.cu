#include <cuda_fp16.h>
#include "GeluPlugin.h"

constexpr float rsqrt_2 = 0.707106781186547524f;
using half              = __half;
using namespace nvinfer1;
using nvinfer1::plugin::GeluPlugin;
using nvinfer1::plugin::GeluPluginCreator;

template <typename T>
__global__ void GeluKernel(const T* input, T* output, const int N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    const T in    = input[index];
    output[index] = 0.5f * in * (1.0f + erff(rsqrt_2 * in));
  }
}

template <>
__global__ void GeluKernel(const half* input, half* output, const int N) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    const half in = input[index];
    output[index] = __float2half(0.5f) * in * __float2half(1.0f + erff(rsqrt_2 * __half2float(in)));
  }
}

int ComputeGelu(cudaStream_t stream, const float* input, float* output, const int N) {
  const int block = 256;
  const int grid  = (N + block - 1) / block;
  GeluKernel<float><<<grid, block, 0, stream>>>(input, output, N);
  return 0;
}

int ComputeGelu(cudaStream_t stream, const half* input, half* output, const int N) {
  const int block = 256;
  // if (0 == (N & 1)) {
  //   const int half_n    = N / 2;
  //   const int grid      = (half_n + block - 1) / block;
  //   const half2* input2 = reinterpret_cast<const half2*>(input);
  //   half2* output2      = reinterpret_cast<half2*>(output);
  //   GeluKernel<half2><<<grid, block, 0, stream>>>(input2, output2, N);
  // } else {
  const int grid = (N + block - 1) / block;
  GeluKernel<half><<<grid, block, 0, stream>>>(input, output, N);
  // }
  return 0;
}

int GeluPlugin::enqueue(const PluginTensorDesc* inputDesc,
                        const PluginTensorDesc* outputDesc,
                        const void* const* inputs,
                        void* const* outputs,
                        void* workspace,
                        cudaStream_t stream) noexcept {
  DataType input_type = inputDesc->type;
  int input_volume    = volume(inputDesc[0].dims);
  if (input_type == DataType::kFLOAT) {
    const float* const input = static_cast<const float*>(inputs[0]);
    float* const output      = static_cast<float*>(outputs[0]);
    ComputeGelu(stream, input, output, input_volume);
  } else if (input_type == DataType::kHALF) {
    const half* const input = static_cast<const half*>(inputs[0]);
    half* const output      = static_cast<half*>(outputs[0]);
    ComputeGelu(stream, input, output, input_volume);
  } else {
    return -1;
  }
  return 0;
}

PluginFieldCollection GeluPluginCreator::fc_{};
std::vector<PluginField> GeluPluginCreator::attrs_{};
REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);
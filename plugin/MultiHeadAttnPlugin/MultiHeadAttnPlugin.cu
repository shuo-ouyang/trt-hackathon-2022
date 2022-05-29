#include "MultiHeadAttnPlugin.h"

using namespace nvinfer1;
using namespace bert;
using nvinfer1::plugin::MHARunner;
using nvinfer1::plugin::MultiHeadAttnPlugin;
using nvinfer1::plugin::MultiHeadAttnPluginCreator;
using nvinfer1::plugin::UnfusedMHARunner;

template <typename T, int TPB, int VPT>
__global__ void
maskedSoftmax(const float rsqrtHeadSize, const T* input, T* output, const int* maskIdx) {
  using BlockReduce = cub::BlockReduce<float, TPB>;

  union SMem {
    T shm[VPT * TPB];
    typename BlockReduce::TempStorage reduce;
    SMem() {}
  };
  __shared__ SMem tmp;

  // grid: (NxS, B)
  const int b           = blockIdx.y;
  const int blockOffset = (b * gridDim.x + blockIdx.x) * TPB;
  __shared__ int lastValid;
  if (threadIdx.x == 0) {
    lastValid = min(TPB, maskIdx[b]);
  }
  __syncthreads();
  float local[VPT];

  __shared__ float rZ;
  __shared__ float fMax[VPT];

  const int idx = (blockOffset + threadIdx.x) * VPT;
  T* myshm      = &tmp.shm[threadIdx.x * VPT];
  copy<sizeof(T) * VPT>(&input[idx], myshm);

  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = (threadIdx.x < lastValid) ? float(tmp.shm[it * TPB + threadIdx.x]) : -FLT_MAX;
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
    if (threadIdx.x == 0) {
      fMax[it] = maxElem;
    }
    __syncthreads();
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] =
        (threadIdx.x < lastValid) ? myExp<float>(rsqrtHeadSize * (local[it] - fMax[it])) : 0.f;
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

    if (threadIdx.x == 0) {
      rZ = (1.f) / Z;
    }
    __syncthreads();
    local[it] *= rZ;
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    tmp.shm[it * TPB + threadIdx.x] = local[it];
  }
  __syncthreads();
  copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, int TPB, int VPT>
__global__ void softmax(const float rsqrtHeadSize, const T* input, T* output) {
  float local[VPT];

  using BlockReduce = cub::BlockReduce<float, TPB>;

  union SMem {
    T shm[VPT * TPB];
    typename BlockReduce::TempStorage reduce;
    SMem() {}
  };
  __shared__ SMem tmp;

  __shared__ float rZ;
  __shared__ float fMax[VPT];

  const int idx = (TPB * blockIdx.x + threadIdx.x) * VPT;
  T* myshm      = &tmp.shm[threadIdx.x * VPT];
  copy<sizeof(T) * VPT>(&input[idx], myshm);

  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = float(tmp.shm[it * TPB + threadIdx.x]);
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    float maxElem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
    if (threadIdx.x == 0) {
      fMax[it] = maxElem;
    }
    __syncthreads();
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = myExp<float>(rsqrtHeadSize * (local[it] - fMax[it]));
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {

    const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

    if (threadIdx.x == 0) {
      rZ = 1.f / Z;
    }
    __syncthreads();
    local[it] *= rZ;
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    tmp.shm[it * TPB + threadIdx.x] = local[it];
  }
  __syncthreads();
  copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, unsigned TPB>
__global__ void
scaledSoftmaxKernelSmall(const int ld, const float rsqrtHeadSize, const T* input, T* output) {
  scaledSoftmaxSmall<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void
scaledSoftmaxKernel(const int ld, const float rsqrtHeadSize, const T* input, T* output) {
  scaledSoftmax<T, TPB>(ld, ld, rsqrtHeadSize, input, output);
}

template <typename T>
int computeScaledSoftmax(cudaStream_t stream,
                         const int ld,
                         const int B,
                         const int N,
                         const float rsqrtHeadSize,
                         const T* input,
                         T* output) {

  constexpr int VPT = 16 / sizeof(T);

  const dim3 grid(ld * N, B, 1);

  if (ld <= 32) {
    const int blockSize = 32;
    scaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
  } else if (ld < 128) {
    const int blockSize = 128;
    scaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
  } else if (ld == 128) {
    const int grid = B * N * ld / (VPT);
    softmax<T, 128, VPT><<<grid, 128, 0, stream>>>(rsqrtHeadSize, input, output);
  }

  else if (ld == 384) {
    const int grid = B * N * ld / (VPT);
    softmax<T, 384, VPT><<<grid, 384, 0, stream>>>(rsqrtHeadSize, input, output);
  } else {
    const int blockSize = 256;

    scaledSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, input, output);
  }

  CHECK(cudaPeekAtLastError());
  return 0;
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernelSmall(const int ld,
                                               const float rsqrtHeadSize,
                                               const int* maskIdx,
                                               const T* input,
                                               T* output) {
  __shared__ int lastValid;

  if (threadIdx.x == 0) {
    lastValid = min(ld, maskIdx[blockIdx.y]);
  }
  __syncthreads();

  scaledSoftmaxSmall<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernel(const int ld,
                                          const float rsqrtHeadSize,
                                          const int* maskIdx,
                                          const T* input,
                                          T* output) {

  __shared__ int lastValid;

  if (threadIdx.x == 0) {
    lastValid = min(ld, maskIdx[blockIdx.y]);
  }
  __syncthreads();
  scaledSoftmax<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T>
int computeMaskedScaledSoftmax(cudaStream_t stream,
                               const int ld,
                               const int B,
                               const int N,
                               const float rsqrtHeadSize,
                               const int* maskIdx,
                               const T* input,
                               T* output) {
  // Mask idx is of length B and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  const dim3 grid(ld * N, B, 1);
  // for smaller problems, e.g. BERT base B=1, this is not optimal
  if (ld <= 32) {
    constexpr int blockSize = 32;
    maskedScaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  } else if (ld < 128) {
    constexpr int blockSize = 128;
    maskedScaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  } else if (ld == 128) {
    if (B == 1) {
      constexpr int VPT       = 4 / sizeof(T);
      constexpr int blockSize = 128;
      const dim3 grid(ld * N / VPT, B, 1);
      maskedSoftmax<T, blockSize, VPT>
          <<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
    } else {
      constexpr int VPT       = 16 / sizeof(T);
      constexpr int blockSize = 128;
      const dim3 grid(ld * N / VPT, B, 1);
      maskedSoftmax<T, blockSize, VPT>
          <<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
    }
  } else if (ld == 384) {
    if (B == 1) {
      constexpr int VPT       = 4 / sizeof(T);
      constexpr int blockSize = 384;
      const dim3 grid(ld * N / VPT, B, 1);
      maskedSoftmax<T, blockSize, VPT>
          <<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
    } else {
      constexpr int VPT       = 16 / sizeof(T);
      constexpr int blockSize = 384;
      const dim3 grid(ld * N / VPT, B, 1);
      maskedSoftmax<T, blockSize, VPT>
          <<<grid, blockSize, 0, stream>>>(rsqrtHeadSize, input, output, maskIdx);
    }
  } else {
    constexpr int blockSize = 256;
    maskedScaledSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  }

  CHECK(cudaPeekAtLastError());
  return 0;
}

std::pair<int, int> tuneBatchedGemm(const int B,
                                    const int S,
                                    const int numHeads,
                                    const int headSize,
                                    const int smVersion) {
  const int nruns = 500;
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasSetStream(cublas, stream);
  cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

  using T             = half;
  const int omatSize  = S * S;
  const int numMats   = B * numHeads;
  const int ldQKV     = 3 * B * numHeads * headSize;
  const int strideQKV = 3 * headSize;
  const int ldOut     = B * numHeads * headSize;
  const int strideOut = headSize;

  const size_t inBytes  = S * B * 3 * numHeads * headSize * sizeof(T);
  const size_t qkBytes  = S * S * B * numHeads * sizeof(T);
  const size_t outBytes = S * B * numHeads * headSize * sizeof(T);

  T* input  = nullptr;
  T* qkptr  = nullptr;
  T* output = nullptr;
  cudaMalloc(&input, inBytes);
  cudaMalloc(&qkptr, qkBytes);
  cudaMalloc(&output, outBytes);
  cudaMemset(input, 1, inBytes);
  cudaMemset(qkptr, 1, qkBytes);

  // input: SxBx3xNxH
  const T* qptr = input;
  const T* kptr = qptr + headSize;
  const T* vptr = kptr + headSize;

  const int startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const int endAlgo   = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  int best1           = startAlgo;
  int best2           = startAlgo;
  float ms1           = 1000000;
  float ms2           = 1000000;

  ASSERT(smVersion >= kSM_53);
  for (int a = startAlgo; a <= endAlgo; a++) {
    cublasGemmAlgo_t algo = static_cast<cublasGemmAlgo_t>(a);
    float ms1_, ms2_;
    // qkptr: BxNxSxS
    cudaEventRecord(start, stream);
    for (int r = 0; r < nruns; r++) {
      CUBLASASSERT(cublasGemmStridedBatchedEx<T>(cublas,
                                                 CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 S,
                                                 S,
                                                 headSize,
                                                 T(1.f),
                                                 kptr,
                                                 ldQKV,
                                                 strideQKV,
                                                 qptr,
                                                 ldQKV,
                                                 strideQKV,
                                                 T(0.f),
                                                 qkptr,
                                                 S,
                                                 omatSize,
                                                 numMats,
                                                 algo));
    }

    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&ms1_, start, stop);
    if (ms1_ < ms1) {
      best1 = algo;
      ms1   = ms1_;
    }

    // pptr: BxNxSxS
    // output: SxBxNxH
    cudaEventRecord(start, stream);
    for (int r = 0; r < nruns; r++) {
      CUBLASASSERT(cublasGemmStridedBatchedEx<T>(cublas,
                                                 CUBLAS_OP_N,
                                                 CUBLAS_OP_N,
                                                 headSize,
                                                 S,
                                                 S,
                                                 1.f,
                                                 vptr,
                                                 ldQKV,
                                                 strideQKV,
                                                 qkptr,
                                                 S,
                                                 omatSize,
                                                 0.f,
                                                 output,
                                                 ldOut,
                                                 strideOut,
                                                 numMats,
                                                 algo));
    }

    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&ms2_, start, stop);

    if (ms2_ < ms2) {
      best2 = algo;
      ms2   = ms2_;
    }
  }

  cudaFree(input);
  cudaFree(qkptr);
  cudaFree(output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublas);
  return std::make_pair(best1, best2);
}

template int computeScaledSoftmax<float>(cudaStream_t stream,
                                         const int ld,
                                         const int B,
                                         const int N,
                                         const float rsqrtHeadSize,
                                         const float* input,
                                         float* output);
template int computeScaledSoftmax<half>(cudaStream_t stream,
                                        const int ld,
                                        const int B,
                                        const int N,
                                        const float rsqrtHeadSize,
                                        const half* input,
                                        half* output);

template int computeMaskedScaledSoftmax<float>(cudaStream_t stream,
                                               const int ld,
                                               const int B,
                                               const int N,
                                               const float rsqrtHeadSize,
                                               const int* maskIdx,
                                               const float* input,
                                               float* output);
template int computeMaskedScaledSoftmax<half>(cudaStream_t stream,
                                              const int ld,
                                              const int B,
                                              const int N,
                                              const float rsqrtHeadSize,
                                              const int* maskIdx,
                                              const half* input,
                                              half* output);

size_t MHARunner::getSerializationSize() const noexcept {
  return sizeof(s_) + sizeof(b_);
}

void MHARunner::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, s_);
  serialize_value(&buffer, b_);
}

void MHARunner::deserialize(const void* data, size_t length) {
  deserialize_value(&data, &length, &s_);
  deserialize_value(&data, &length, &b_);
  setup(s_, b_);
}

UnfusedMHARunner::UnfusedMHARunner(const nvinfer1::DataType type,
                                   const int numHeads,
                                   const int headSize,
                                   const int sm)
    : MHARunner(type, numHeads, headSize), is_best_algo_found_(false),
      algo_batched_ex1_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      algo_batched_ex2_(CUBLAS_GEMM_DEFAULT_TENSOR_OP), sm_(sm) {
  CUBLASASSERT(cublasCreate(&cublas_));
}

UnfusedMHARunner::~UnfusedMHARunner() {
  CUBLASASSERT(cublasDestroy(cublas_));
}

size_t UnfusedMHARunner::getSerializationSize() const noexcept {
  return sizeof(algo_batched_ex1_) + sizeof(algo_batched_ex2_) + MHARunner::getSerializationSize();
}

void UnfusedMHARunner::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, algo_batched_ex1_);
  serialize_value(&buffer, algo_batched_ex2_);
  MHARunner::serialize(buffer);
}

void UnfusedMHARunner::deserialize(const void* data, size_t length) {
  is_best_algo_found_ = true;
  deserialize_value(&data, &length, &algo_batched_ex1_);
  deserialize_value(&data, &length, &algo_batched_ex2_);
  MHARunner::deserialize(data, length);
}

void UnfusedMHARunner::setup(const int S, const int B) {
  MHARunner::setup(S, B);
  if (type_ == DataType::kHALF && !is_best_algo_found_) {
    std::tie(algo_batched_ex1_, algo_batched_ex2_) =
        tuneBatchedGemm(B, S, num_heads_, head_size_, sm_);
    is_best_algo_found_ = true;

    BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 1 for batch gemms: ", algo_batched_ex1_);
    BERT_DEBUG_VALUE("QKV Plugin - Selected Algo 2 for batch gemms: ", algo_batched_ex2_);
  }
}
bool UnfusedMHARunner::isValid(int s) const {
  return type_ != DataType::kINT8;
}

size_t UnfusedMHARunner::getWorkspaceSize() const {
  return 2UL * word_size_ * omat_size_ * num_mats_;
}

void UnfusedMHARunner::run(const PluginTensorDesc* inputDesc,
                           const PluginTensorDesc* outputDesc,
                           const void* const* inputs,
                           void* const* outputs,
                           void* workspace,
                           cudaStream_t stream) {
  this->run(inputDesc[0], outputDesc[0], inputs[0], inputs[1], outputs[0], workspace, stream);
}

void UnfusedMHARunner::run(const PluginTensorDesc& inputDesc,
                           const PluginTensorDesc& outputDesc,
                           const void* qkvPtr,
                           const void* maskPtr,
                           void* output,
                           void* workspace,
                           cudaStream_t stream) {
  const int* maskIdx = static_cast<const int*>(maskPtr);

  cublasSetStream(cublas_, stream);

  // Q, K, V: BxNxSxH (inputs)
  // Q * K': BxNxSxS (-> scratch1)
  // P: BxNxSxS (-> scratch2)
  // P * V: BxNxSxH (output)

  if (type_ == DataType::kHALF) {
    CublasConfigHelper helper(cublas_);
    const half* qptr = static_cast<const half*>(qkvPtr);
    const half* kptr = qptr + head_size_;
    const half* vptr = kptr + head_size_;
    half* qkptr      = static_cast<half*>(workspace);
    half* pptr       = qkptr + omat_size_ * num_mats_;
    half alpha       = 1.f;
    half beta        = 0.f;
    cublasGemmStridedBatchedEx(cublas_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               s_,
                               s_,
                               head_size_,
                               &alpha,
                               kptr,
                               CUDA_R_16F,
                               ld_qkv_,
                               stride_qkv_,
                               qptr,
                               CUDA_R_16F,
                               ld_qkv_,
                               stride_qkv_,
                               &beta,
                               qkptr,
                               CUDA_R_16F,
                               s_,
                               omat_size_,
                               num_mats_,
                               CUDA_R_16F,
                               static_cast<cublasGemmAlgo_t>(algo_batched_ex1_));

    // apply softmax
    if (maskIdx) { // if we have a mask
      computeMaskedScaledSoftmax<half>(
          stream, s_, b_, num_heads_, rsqrt_head_size_, maskIdx, qkptr, pptr);
    } else { // if we don't have a mask
      computeScaledSoftmax<half>(stream, s_, b_, num_heads_, rsqrt_head_size_, qkptr, pptr);
    }

    // compute P*V (as V*P)
    cublasGemmStridedBatchedEx(cublas_,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               head_size_,
                               s_,
                               s_,
                               &alpha,
                               vptr,
                               CUDA_R_16F,
                               ld_qkv_,
                               stride_qkv_,
                               pptr,
                               CUDA_R_16F,
                               s_,
                               omat_size_,
                               &beta,
                               output,
                               CUDA_R_16F,
                               ld_out_,
                               stride_out_,
                               num_mats_,
                               CUDA_R_16F,
                               static_cast<cublasGemmAlgo_t>(algo_batched_ex2_));
  } else {
    const float* qptr = static_cast<const float*>(qkvPtr);
    const float* kptr = qptr + head_size_;
    const float* vptr = kptr + head_size_;
    float* qkptr      = static_cast<float*>(workspace);
    float* pptr       = qkptr + omat_size_ * num_mats_;
    float* outptr     = static_cast<float*>(output);
    CUBLASASSERT(cublasGemmStridedBatched<float>(cublas_,
                                                 CUBLAS_OP_T,
                                                 CUBLAS_OP_N,
                                                 s_,
                                                 s_,
                                                 head_size_,
                                                 1.f,
                                                 kptr,
                                                 ld_qkv_,
                                                 stride_qkv_,
                                                 qptr,
                                                 ld_qkv_,
                                                 stride_qkv_,
                                                 0.f,
                                                 qkptr,
                                                 s_,
                                                 omat_size_,
                                                 num_mats_));

    // apply softmax
    if (maskIdx) { // if we have a mask
      computeMaskedScaledSoftmax<float>(
          stream, s_, b_, num_heads_, rsqrt_head_size_, maskIdx, qkptr, pptr);
    } else { // if we don't have a mask
      computeScaledSoftmax<float>(stream, s_, b_, num_heads_, rsqrt_head_size_, qkptr, pptr);
    }

    CUBLASASSERT(cublasGemmStridedBatched<float>(cublas_,
                                                 CUBLAS_OP_N,
                                                 CUBLAS_OP_N,
                                                 head_size_,
                                                 s_,
                                                 s_,
                                                 1.f,
                                                 vptr,
                                                 ld_qkv_,
                                                 stride_qkv_,
                                                 pptr,
                                                 s_,
                                                 omat_size_,
                                                 0.f,
                                                 outptr,
                                                 ld_out_,
                                                 stride_out_,
                                                 num_mats_));
  }
}

int32_t MultiHeadAttnPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                     const PluginTensorDesc* outputDesc,
                                     const void* const* inputs,
                                     void* const* outputs,
                                     void* workspace,
                                     cudaStream_t stream) noexcept {
  ASSERT(s_ == inputDesc->dims.d[SDIM]);
  ASSERT(b_ == inputDesc->dims.d[BDIM]);

  const void* mask_ptr = has_mask_ ? inputs[1] : nullptr;
  if (fused_dispatcher_.get() and fused_dispatcher_->isValid(inputDesc->dims.d[SDIM])) {
    fused_dispatcher_->run(
        inputDesc[0], outputDesc[0], inputs[0], mask_ptr, outputs[0], workspace, stream);
  } else {
    unfused_dispatcher_->run(
        inputDesc[0], outputDesc[0], inputs[0], mask_ptr, outputs[0], workspace, stream);
  }
  CHECK(cudaPeekAtLastError());
  return 0;
}
PluginFieldCollection MultiHeadAttnPluginCreator::fc_{};
std::vector<PluginField> MultiHeadAttnPluginCreator::attrs_;

REGISTER_TENSORRT_PLUGIN(MultiHeadAttnPluginCreator);
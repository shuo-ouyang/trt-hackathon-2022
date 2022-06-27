#include <NvInfer.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <numeric>
#include <string>
#include <vector>

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (((X) + (Y)-1) / (Y) * (Y))
#define CHECK_CUDA(call)                                                                           \
  {                                                                                                \
    const cudaError_t error = call;                                                                \
    if (error != cudaSuccess) {                                                                    \
      printf("Error: %s:%d\n", __FILE__, __LINE__);                                                \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));                          \
      exit(1);                                                                                     \
    }                                                                                              \
  }
// #define DEBUG_FUNC()                                                                               \
//   { printf("[DEBUG] %s, %s:%d\n", __FILE__, __FUNCTION__, __LINE__); }

#define DEBUG_FUNC()                                                                               \
  {}

#define ALIGN_TO(X, Y) (CEIL_DIVIDE(X, Y) * (Y))

inline void check(cudaError_t ret, int line) {
  if (ret != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << ", line: " << line << std::endl;
  }
}

#define CHECK(_x) check((_x), __LINE__)

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
  using type = uint16_t;
};
template <>
struct BytesToType<4> {
  using type = uint32_t;
};
template <>
struct BytesToType<8> {
  using type = uint64_t;
};
template <>
struct BytesToType<16> {
  using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data) {
  using T = typename BytesToType<Bytes>::type;

  const T* in = static_cast<const T*>(local);
  T* out      = static_cast<T*>(data);
  *out        = *in;
}

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T>
struct mySum {
  __host__ __device__ __forceinline__ kvp<T> operator()(const kvp<T>& a, const kvp<T>& b) const {
    return kvp<T>(a.key + b.key, a.value + b.value);
  }
};

namespace {
static const char* PLUGIN_NAME{"LayerNorm"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
namespace nvinfer1 {
namespace plugin {

class LayerNormPlugin final : public IPluginV2DynamicExt {
public:
  LayerNormPlugin(const std::string& name) : name_(name) {
    DEBUG_FUNC();
  }

  LayerNormPlugin(const std::string& name, const void* data, size_t length) : name_(name) {
    DEBUG_FUNC();
  }

  LayerNormPlugin() = delete;

  ~LayerNormPlugin() {
    DEBUG_FUNC();
  }

  size_t getSerializationSize() const noexcept override {
    DEBUG_FUNC();
    return 0;
  }

  void serialize(void* buffer) const noexcept override {
    DEBUG_FUNC();
  }

  IPluginV2DynamicExt* clone() const noexcept override {
    DEBUG_FUNC();
    return new LayerNormPlugin(name_);
  }

  int getNbOutputs() const noexcept override {
    DEBUG_FUNC();
    return 1;
  }

  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) noexcept override {
    DEBUG_FUNC();
    return inputs[0];
  }

  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {

    switch (pos) {
    case 0:
      return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) &&
                 inOut[0].format == TensorFormat::kLINEAR ||
             inOut[0].type == DataType::kINT8 && (inOut[0].format == TensorFormat::kCHW4 ||
                                                  inOut[0].format == TensorFormat::kCHW32);
    case 1:
    case 2:
      return inOut[pos].type == inOut[0].type ||
             inOut[0].type == DataType::kINT8 && inOut[pos].type == DataType::kHALF;
    case 3:
      return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    default: // should NOT be here!
      return false;
    }
    return false;
  }

  DataType getOutputDataType(int outputIndex,
                             const DataType* inputTypes,
                             int nbInputs) const noexcept override {
    DEBUG_FUNC();
    return inputTypes[0];
  }

  void configurePlugin(const DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) noexcept override {
    DEBUG_FUNC();
  }

  size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const PluginTensorDesc* outputs,
                          int32_t nbOutputs) const noexcept override {
    DEBUG_FUNC();
    return 0;
  }

  void setPluginNamespace(const char* szNamespace) noexcept override {
    DEBUG_FUNC();
    namespace_ = szNamespace;
  }
  const char* getPluginNamespace() const noexcept override {
    DEBUG_FUNC();
    return namespace_.c_str();
  }
  const char* getPluginType() const noexcept override {
    DEBUG_FUNC();
    return PLUGIN_NAME;
  }
  const char* getPluginVersion() const noexcept override {
    DEBUG_FUNC();
    return PLUGIN_VERSION;
  }
  int initialize() noexcept override {
    DEBUG_FUNC();
    return 0;
  }
  void terminate() noexcept override {
    DEBUG_FUNC();
    return;
  }

  void destroy() noexcept override {
    delete this;
  }

  int enqueue(const PluginTensorDesc* inputDesc,
              const PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) noexcept override;

private:
  std::string name_;
  std::string namespace_;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator {
private:
  static PluginFieldCollection fc_;
  static std::vector<PluginField> attrs_;
  std::string namespace_;

public:
  LayerNormPluginCreator() {}

  ~LayerNormPluginCreator() {}

  IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
    DEBUG_FUNC();
    return new LayerNormPlugin(name);
  }

  IPluginV2* deserializePlugin(const char* name,
                               const void* serialData,
                               size_t serialLength) noexcept override {
    DEBUG_FUNC();
    return new LayerNormPlugin(name, serialData, serialLength);
  }

  void setPluginNamespace(const char* szNamespace) noexcept override {
    namespace_ = szNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

  const char* getPluginName() const noexcept override {
    return PLUGIN_NAME;
  }

  const char* getPluginVersion() const noexcept override {
    return PLUGIN_VERSION;
  }

  const PluginFieldCollection* getFieldNames() noexcept override {
    return &fc_;
  }
}; // class LayerNormPluginCreator

} // namespace plugin

} // namespace nvinfer1
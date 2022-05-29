#include <NvInfer.h>

#include <algorithm>
#include <cub/cub.cuh>
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
    DEBUG_FUNC();
    if (inOut[pos].format != TensorFormat::kLINEAR) {
      return false;
    }

    bool res = false;
    switch (pos) {
    case 0:
      res = inOut[pos].type == DataType::kFLOAT or inOut[pos].type == DataType::kHALF;
      break;
    case 1:
      res = inOut[pos].type == DataType::kFLOAT;
      break;
    case 2:
      res = inOut[pos].type == DataType::kFLOAT;
      break;
    case 3:
      res = inOut[pos].type == DataType::kFLOAT or inOut[pos].type == DataType::kHALF;
      break;
    default: // should NOT be here
      break;
    }
    return res;
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
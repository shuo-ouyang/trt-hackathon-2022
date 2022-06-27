#include <NvInfer.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace {
static const char* PLUGIN_NAME{"Gelu"};
static const char* PLUGIN_VERSION{"1"};
}  // namespace

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

namespace nvinfer1 {
namespace plugin {
class GeluPlugin final : public IPluginV2DynamicExt {
 public:
  GeluPlugin(const std::string& name) : name_(name) {}

  GeluPlugin(const std::string& name, const void* data, size_t length) : name_(name) {}

  GeluPlugin() = delete;

  ~GeluPlugin() {}

  size_t getSerializationSize() const noexcept {
    return 0;
  }

  void serialize(void* buffer) const noexcept override {
    return;
  }

  IPluginV2DynamicExt* clone() const noexcept override {
    return new GeluPlugin(name_);
  }

  int getNbOutputs() const noexcept override {
    return 1;
  }

  DimsExprs getOutputDimensions(int32_t outputIndex,
                                const DimsExprs* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) noexcept override {
    return inputs[0];
  }

  bool supportsFormatCombination(int32_t pos,
                                 const PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (inOut[pos].format != TensorFormat::kLINEAR) {
      return false;
    }
    bool res = false;
    switch (pos) {
      case 0:
        res = inOut[pos].type == DataType::kFLOAT or inOut[pos].type == DataType::kHALF;
        break;
      case 1:
        res = inOut[pos].type == DataType::kFLOAT or inOut[pos].type == DataType::kHALF;
        break;
      default:
        break;
    }
    return res;
  }

  DataType getOutputDataType(int outputIdex,
                             const DataType* inputTypes,
                             int nbInputs) const noexcept override {
    return inputTypes[0];
  }

  void configurePlugin(const DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) noexcept override {
    return;
  }

  size_t getWorkspaceSize(const PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const PluginTensorDesc* outputs,
                          int32_t nbOutputs) const noexcept override {
    return 0;
  }

  void setPluginNamespace(const char* szNamespace) noexcept override {
    namespace_ = szNamespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }
  const char* getPluginType() const noexcept override {
    return PLUGIN_NAME;
  }
  const char* getPluginVersion() const noexcept override {
    return PLUGIN_VERSION;
  }
  int initialize() noexcept override {
    return 0;
  }
  void terminate() noexcept override {
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
};

class GeluPluginCreator : public IPluginCreator {
 public:
  GeluPluginCreator() {}

  ~GeluPluginCreator() {}

  IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
    return new GeluPlugin(name);
  }

  IPluginV2* deserializePlugin(const char* name,
                               const void* serialData,
                               size_t serialLength) noexcept override {
    return new GeluPlugin(name, serialData, serialLength);
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

 private:
  static PluginFieldCollection fc_;
  static std::vector<PluginField> attrs_;
  std::string namespace_;
};  // class GeluPluginCreator

}  // namespace plugin
}  // namespace nvinfer1
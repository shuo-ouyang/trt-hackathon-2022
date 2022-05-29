
#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include <cuda.h>
#include "cublas_v2.h"
#include "NvInferPlugin.h"
#include "serialize.hpp"
#include "common.cuh"
#include "bertCommon.h"

#undef ASSERT
#define ASSERT(condition)                                                                          \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::cout << "Assertion failure: " << #condition << std::endl;                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)
#define TRT_UNUSED (void)

namespace {
static const char* PLUGIN_NAME{"MultiHeadAttn"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

using namespace bert;
constexpr uint32_t IIDX = 0; // index of the input tensor
constexpr uint32_t MIDX = 1; // index of the mask

namespace nvinfer1 {
namespace plugin {

class MHARunner {
public:
  MHARunner(const nvinfer1::DataType type, const int32_t num_heads, const int32_t head_size)
      : type_(type), s_(0), b_(0), omat_size_(0), num_mats_(0), num_heads_(num_heads),
        head_size_(head_size), word_size_(getElementSize(type)), ld_qkv_(0), stride_qkv_(0),
        ld_out_(0), stride_out_(0), rsqrt_head_size_(1.F / sqrtf(head_size)) {}
  virtual ~MHARunner() = default;

  virtual void setup(const int32_t S, const int32_t B) {
    ASSERT(S);
    ASSERT(B);

    b_          = B;
    s_          = S;
    ld_qkv_     = 3 * B * num_heads_ * head_size_;
    stride_qkv_ = 3 * head_size_;

    ld_out_     = B * num_heads_ * head_size_;
    stride_out_ = head_size_;
    omat_size_  = S * S;
    num_mats_   = B * num_heads_;
  }

  virtual void run(const nvinfer1::PluginTensorDesc& inputDesc,
                   const nvinfer1::PluginTensorDesc& outputDesc,
                   const void* qkvPtr,
                   const void* maskPtr,
                   void* output,
                   void* workspace,
                   cudaStream_t stream) = 0;

  virtual void run(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream) = 0;

  virtual size_t getSerializationSize() const noexcept;
  virtual void serialize(void* buffer) const noexcept;
  virtual void deserialize(const void* data, size_t length);

  virtual size_t getWorkspaceSize() const = 0;

  virtual bool isValid(int32_t s) const = 0;

protected:
  nvinfer1::DataType type_;

  int32_t s_;
  int32_t b_;
  int32_t omat_size_;
  int32_t num_mats_;
  int32_t num_heads_;
  int32_t head_size_;
  int32_t word_size_;
  int32_t ld_qkv_;
  int32_t stride_qkv_;
  int32_t ld_out_;
  int32_t stride_out_;

  float rsqrt_head_size_;
};

class UnfusedMHARunner : public MHARunner {
public:
  UnfusedMHARunner(const nvinfer1::DataType type,
                   const int32_t num_heads,
                   const int32_t head_size,
                   const int32_t sm);
  virtual ~UnfusedMHARunner();

  virtual void setup(const int32_t S, const int32_t B) override;

  void run(const nvinfer1::PluginTensorDesc& inputDesc,
           const nvinfer1::PluginTensorDesc& outputDesc,
           const void* qkvPtr,
           const void* maskPtr,
           void* output,
           void* workspace,
           cudaStream_t stream) override;

  void run(const nvinfer1::PluginTensorDesc* inputDesc,
           const nvinfer1::PluginTensorDesc* outputDesc,
           const void* const* inputs,
           void* const* outputs,
           void* workspace,
           cudaStream_t stream) override;

  size_t getWorkspaceSize() const override;

  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void deserialize(const void* data, size_t length) override;
  bool isValid(int32_t s) const override;

private:
  bool is_best_algo_found_;
  int32_t algo_batched_ex1_;
  int32_t algo_batched_ex2_;
  cublasHandle_t cublas_;
  int32_t sm_;
};

class MultiHeadAttnPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
  MultiHeadAttnPlugin(const std::string name,
                      const nvinfer1::DataType type,
                      const int32_t hidden_size,
                      const int32_t num_heads,
                      bool has_mask = false)
      : layer_name_(name), s_(0), b_(0), head_size_(hidden_size / num_heads),
        hidden_size_(hidden_size), num_heads_(num_heads), has_mask_(has_mask), type_(type) {
    sm_ = getSMVersion();
  }

  MultiHeadAttnPlugin(const std::string name, const void* data, size_t length) : layer_name_(name) {
    deserialize_value(&data, &length, &type_);
    deserialize_value(&data, &length, &num_heads_);
    deserialize_value(&data, &length, &head_size_);
    deserialize_value(&data, &length, &has_mask_);
    deserialize_value(&data, &length, &hidden_size_);
    deserialize_value(&data, &length, &sm_);
    deserialize_value(&data, &length, &s_);
    deserialize_value(&data, &length, &b_);

    createMHARunner();

    int32_t has_unfused_runner = 1;
    deserialize_value(&data, &length, &has_unfused_runner);
    if (has_unfused_runner) {
      ASSERT(unfused_dispatcher_.get());
      unfused_dispatcher_->deserialize(data, length);
    }
  }

  MultiHeadAttnPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
    MultiHeadAttnPlugin* ret = nullptr;
    // the workspacesize is 0 if we have not call setup the dispatcher yet.
    if (unfused_dispatcher_.get() && unfused_dispatcher_->getWorkspaceSize()) {
      std::vector<char> buff;
      buff.resize(getSerializationSize());
      serialize(buff.data());

      ret = new MultiHeadAttnPlugin(layer_name_, buff.data(), buff.size());
    } else {
      ret = new MultiHeadAttnPlugin(layer_name_, type_, hidden_size_, num_heads_, has_mask_);
    }

    ret->setPluginNamespace(name_space_.c_str());
    return ret;
  }
  nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
                                          const nvinfer1::DimsExprs* inputs,
                                          int32_t nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override {
    // Input is BxSx3*N*H, output should be BxSxN*H
    ASSERT(outputIndex == 0);
    // Copy over everything
    nvinfer1::DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    const auto* three = exprBuilder.constant(3);
    output.d[HDIM] =
        exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[HDIM], *three);
    return output;
  }

  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    ASSERT(pos >= 0);
    ASSERT(pos < 2 + has_mask_);
    ASSERT(nbInputs == 1 + has_mask_);

    const auto* in  = inOut;
    const auto* out = inOut + nbInputs;
    // only supoort unfused fp32 mha now
    int32_t packed_size = 1;

    if (pos == 0) {
      return in->type == type_ and in->format == TensorFormat::kLINEAR and in->dims.nbDims == 5 and
             (in->dims.d[HDIM] % 3U) == 0 and in->dims.d[3] == 1 and in->dims.d[4] == 1;
    } else {
      // pos 1 is the mask
      if (has_mask_ and pos == 1) {
        const auto* in_mask = &inOut[1];
        return in_mask->type == DataType::kFLOAT and in_mask->format == TensorFormat::kLINEAR and
               in_mask->dims.nbDims == 2 and in_mask->dims.d[0] == in->dims.d[BDIM];
      }
      // output pos
      if (not has_mask_ or pos == 2) {
        return in->type == out->type and out->format == TensorFormat::kLINEAR and
               out->dims.nbDims == 5 and (in->dims.d[HDIM] / 3) == out->dims.d[HDIM] and
               out->dims.d[3] == 1 and out->dims.d[4] == 1 and
               out->dims.d[BDIM] == in->dims.d[BDIM] and out->dims.d[SDIM] == in->dims.d[SDIM];
      }
    }
    return false;
  }

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) noexcept override {
    ASSERT(nbInputs == 1 + has_mask_);
    ASSERT(nbOutputs == 1);
    const PluginTensorDesc& in_desc = in[IIDX].desc;
    TRT_UNUSED in_desc;
    const PluginTensorDesc& out_desc = out->desc;
    TRT_UNUSED out_desc;
    ASSERT(type_ == in_desc.type);
    ASSERT(type_ == out_desc.type);
    ASSERT(in_desc.dims.d[BDIM] == out_desc.dims.d[BDIM]);
    ASSERT(in_desc.dims.d[SDIM] == out_desc.dims.d[SDIM]);
    ASSERT(in_desc.dims.d[HDIM] == 3 * out_desc.dims.d[HDIM]);
    if (has_mask_) {
      const PluginTensorDesc& mask_desc = in[MIDX].desc;
      TRT_UNUSED mask_desc;
      ASSERT(mask_desc.dims.d[0] == in_desc.dims.d[BDIM]);
    }
    createMHARunner();

    const int32_t S = in_desc.dims.d[SDIM];
    const int32_t B = in_desc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : in_desc.dims.d[BDIM];

    if (S <= 0) {
      // in dynamic shape build stage, we setup with max sequence that cannot fused
      const int32_t s_min = in->min.d[SDIM];
      const int32_t s_max = in->max.d[SDIM];
      if (fused_dispatcher_.get()) {
        for (int32_t i = s_max; i >= s_min; --i) {
          if (!fused_dispatcher_->isValid(i)) {
            unfused_dispatcher_->setup(i, B);
            s_ = i;
            b_ = B;
            break;
          }
        }
      } else {
        unfused_dispatcher_->setup(s_max, B);
        s_ = s_max;
        b_ = B;
      }
    } else {
      // in inference stage or in static shape build stage
      if (fused_dispatcher_.get() && fused_dispatcher_->isValid(S)) {
        fused_dispatcher_->setup(S, B);
      } else {
        unfused_dispatcher_->setup(S, B);
      }
      s_ = S;
      b_ = B;
    }
  }

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t nbOutputs) const noexcept override {
    // only unfused kernel need workspace, and we need larger workspace for larger sequence length
    // we have already setup unfused dispatcher with max sequence in configurePlugin
    // if unfused dispatcher is not initialized in configurePlugin
    ASSERT(unfused_dispatcher_.get());
    return unfused_dispatcher_->getWorkspaceSize();
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                  const nvinfer1::PluginTensorDesc* outputDesc,
                  const void* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int32_t index,
                                       const nvinfer1::DataType* inputTypes,
                                       int32_t nbInputs) const noexcept override {
    ASSERT(index == 0);
    ASSERT(inputTypes[0] == DataType::kFLOAT or inputTypes[0] == DataType::kHALF or
           inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
  }

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override {
    return PLUGIN_NAME;
  }

  const char* getPluginVersion() const noexcept override {
    return PLUGIN_VERSION;
  }

  int32_t getNbOutputs() const noexcept override {
    return 1;
  }

  int32_t initialize() noexcept override {
    return 0;
  }

  void terminate() noexcept override {}

  size_t getSerializationSize() const noexcept override {
    ASSERT(unfused_dispatcher_.get());
    return sizeof(num_heads_) + sizeof(head_size_) + sizeof(DataType) + sizeof(has_mask_) +
           sizeof(hidden_size_) + sizeof(sm_) + sizeof(s_) + sizeof(b_) + sizeof(int) +
           unfused_dispatcher_->getSerializationSize();
  }

  void serialize(void* buffer) const noexcept override {
    serialize_value(&buffer, type_);
    serialize_value(&buffer, num_heads_);
    serialize_value(&buffer, head_size_);
    serialize_value(&buffer, has_mask_);
    serialize_value(&buffer, hidden_size_);
    serialize_value(&buffer, sm_);
    serialize_value(&buffer, s_);
    serialize_value(&buffer, b_);

    if (unfused_dispatcher_.get() && unfused_dispatcher_->getWorkspaceSize()) {
      int32_t has_unfused_runner = 1;
      serialize_value(&buffer, has_unfused_runner);
      unfused_dispatcher_->serialize(buffer);
    } else {
      int32_t has_unfused_runner = 0;
      serialize_value(&buffer, has_unfused_runner);
    }
  }

  void destroy() noexcept override {
    delete this;
  }

  void setPluginNamespace(const char* libNamespace) noexcept override {
    name_space_ = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return name_space_.c_str();
  }

protected:
  void createMHARunner() {
    // if (!fused_dispatcher_.get()) {
    //   if (mType == DataType::kHALF) {
    //     fused_dispatcher_.reset(new FusedMHARunnerFP16(mNumHeads, mHeadSize, mSM));
    //   } else if (mType == DataType::kINT8) {
    //     fused_dispatcher_.reset(new FusedMHARunnerInt8(mNumHeads, mHeadSize, mSM, mDqProbs));
    //   }
    // }

    if (!unfused_dispatcher_.get()) {
      unfused_dispatcher_.reset(new UnfusedMHARunner(type_, num_heads_, head_size_, sm_));
    }
  }

private:
  const std::string layer_name_;
  std::string name_space_;

  std::unique_ptr<MHARunner> fused_dispatcher_;
  std::unique_ptr<MHARunner> unfused_dispatcher_;

  int32_t s_;
  int32_t b_;
  int32_t sm_;
  int32_t num_heads_;
  int32_t head_size_;
  int32_t hidden_size_;
  bool has_mask_;
  nvinfer1::DataType type_;
};

class MultiHeadAttnPluginCreator : public nvinfer1::IPluginCreator {
public:
  MultiHeadAttnPluginCreator() {
    attrs_.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    attrs_.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    attrs_.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));

    fc_.nbFields = attrs_.size();
    fc_.fields   = attrs_.data();
  }

  const char* getPluginName() const noexcept override {
    return PLUGIN_NAME;
  }

  const char* getPluginVersion() const noexcept override {
    return PLUGIN_VERSION;
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &fc_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc) noexcept override {
    int32_t hidden_size = 0;
    int32_t num_heads   = 0;
    bool has_mask       = false;
    int32_t type_id     = -1;

    for (int32_t i = 0; i < fc->nbFields; i++) {
      std::string field_name(fc->fields[i].name);
      if (field_name.compare("type_id") == 0) {
        type_id = *static_cast<const int*>(fc->fields[i].data);
      }
      if (field_name.compare("hidden_size") == 0) {
        hidden_size = *static_cast<const int*>(fc->fields[i].data);
      }
      if (field_name.compare("num_heads") == 0) {
        num_heads = *static_cast<const int*>(fc->fields[i].data);
      }
      if (field_name.compare("has_mask") == 0) {
        has_mask = *static_cast<const bool*>(fc->fields[i].data);
      }
    }
    if (type_id < 0 or type_id > 3 or hidden_size <= 0 or num_heads <= 0) {
      return nullptr;
    }
    DataType type = static_cast<DataType>(type_id);

    MultiHeadAttnPlugin* p = new MultiHeadAttnPlugin(name, type, hidden_size, num_heads, has_mask);
    return p;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) noexcept override {
    return new MultiHeadAttnPlugin(name, serialData, serialLength);
  }

  void setPluginNamespace(const char* libNamespace) noexcept {
    name_space_ = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return name_space_.c_str();
  }

private:
  static nvinfer1::PluginFieldCollection fc_;
  static std::vector<nvinfer1::PluginField> attrs_;
  std::string name_space_;
};
} // namespace plugin
} // namespace nvinfer1
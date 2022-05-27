# TensorRT Hackathon 2022

## 总述
---
### 原始模型名称及链接
模型：**UniFormer**(**Uni**fied trans**Former**)

论文：https://arxiv.org/pdf/2201.09450.pdf

代码：https://github.com/Sense-X/UniFormer
### 优化效果

### 运行步骤


## 原始模型
---
### 模型简介
对于图像和视频上的representation learning而言，目前存在两大痛点：
- local redundancy: 视觉数据在局部空间/时间/时空邻域具有相似性，这种局部性质容易引入大量低效的计算。
- global dependency: 要实现准确的识别，需要动态地将不同区域中的目标关联，建模长时依赖。

现有的两大主流模型CNN和ViT，往往只关注解决以上部分问题。Convolution只在局部小邻域聚合上下文，天然地避免了冗余的全局计算，但受限的感受野也难以建模全局依赖。而self-attention通过比较全局相似度，将长距离目标关联，但ViT在浅层编码局部特征十分低效。相较而言，convolution在提取这些浅层特征时，无论是在效果上还是计算量上都具有显著的优势。那么为何不针对网络不同层特征的差异，设计不同的特征学习算子，从而将convolution和self-attention有机地结合起来，各取所长、物尽其用呢？

UniFormer以Transformer的风格，有机地统一convolution和self-attention，发挥二者的优势，同时解决local redundancy和global dependency两大问题，从而实现高效的特征学习。

![a](./figures/framework.png)

模型整体框架如上图所示。Uniformer借鉴了CNN的层次化设计，每层包含多个Transformer风格的UniFormer Block。每个UniFormer Block都由三部分组成：动态位置编码DPE、多头关系聚合器MHRA以及前馈层FFN。

### 模型优化的难点

相比于传统的CNN模型（如ResNet50），基于Transformer的模型在控制FLOPS的同时能够取得不错的精度优势。但是，限制一个模型能否商用落地更多的是延迟（Latency）。

| 模型        | 精度    | FP32/FP16推理延迟（倍数关系） | FP32/FP16显存占用 |
|:---------:|:-----:|:-------------------:|:-------------:|
| ResNet50  | 79.1% | 3/1                 | 636MB/579MB   |
| Uniformer | 82.9% | 4.8/3               | 742MB/676MB   |

从上表[1]中可以看到，Uniformer在FP32和FP16下的推理延迟都高于ResNet50，而且Uniformer对量化不友好，FP16的推理延迟只有FP32的62.5%。因此，本次复赛的目标是在保证精度的情况下，基于TensorRT优化Uniformer的推理延迟。

## 优化过程
---

## 精度与加速效果
---


## References
>[1] https://bbs.huaweicloud.com/blogs/327738

import os
from cuda import cudart
import tensorrt as trt
import numpy as np
import ctypes
import torch
from timm.utils import accuracy
from test_pytorch_accuracy import build_dataset, build_transform
from utils import MetricLogger
import argparse



plan_file = './uf_static_bs32_fp16.plan'

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')

ctypes.cdll.LoadLibrary('../plugins/LayerNormPlugin/LayerNorm.so')
ctypes.cdll.LoadLibrary('../plugins/GeluPlugin/Gelu.so')

if os.path.isfile(plan_file):
    with open(plan_file, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"Failed loading {plan_file}")
        exit()
    print(f"Succeeded loading {plan_file}")
else:
    print(f"Failed finding {plan_file}")
    exit()

nbInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nbOutput = engine.num_bindings - nbInput
context = engine.create_execution_context()
nbChannel = 3
height = 224
width = 224

batch_size = 32

context.set_binding_shape(0, (batch_size, nbChannel, height, width))

hostBuf = []
for i in range(0, nbInput + nbOutput):
    hostBuf.append(np.ones(context.get_binding_shape(i),
                           dtype=trt.nptype(engine.get_binding_dtype(i))))


devBuf = []
for i in range(nbInput + nbOutput):
    devBuf.append(cudart.cudaMalloc(hostBuf[i].nbytes)[1])


transform = build_transform()
dataset, _ = build_dataset('/imagenet/val', transform)
sampler = torch.utils.data.SequentialSampler(dataset)
data_loader_val = torch.utils.data.DataLoader(
    dataset, sampler=sampler,
    batch_size=int(batch_size),
    num_workers=1,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)

metric_logger = MetricLogger(delimiter=" ")

# warmup
for _ in range(30):
    context.execute_v2(devBuf)

for images, labels in metric_logger.log_every(data_loader_val, 10, 'TensorRT: '):
    hostBuf[0] = images.numpy().reshape(-1)
    cudart.cudaMemcpy(devBuf[0], hostBuf[0].ctypes.data, hostBuf[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(devBuf)
    cudart.cudaMemcpy(hostBuf[1].ctypes.data, devBuf[1], hostBuf[1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    acc1, acc5 = accuracy(torch.Tensor(hostBuf[1][:labels.shape[0]].reshape(-1, 1000)), labels, topk=(1, 5))
    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
metric_logger.synchronize_between_processes()
print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
      .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
print(f"Accuracy of the network on the {len(dataset)} test images: {test_stats['acc1']:.1f}%")

for i in range(nbInput + nbOutput):
    cudart.cudaFree(devBuf[i])

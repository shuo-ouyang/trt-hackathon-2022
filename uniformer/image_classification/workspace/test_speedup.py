import os
import sys
import ctypes
import torch
import numpy as np
from glob import glob
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
import sys
sys.path.append('../')
from models.uniformer import uniformer_small

data_file = "./data/"
plan_file = './uf.plan'

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')

ctypes.cdll.LoadLibrary('../../../plugin/LayerNormPlugin/LayerNorm.so')
ctypes.cdll.LoadLibrary('../../../plugin/GeluPlugin/Gelu.so')


def load_engine(plan_file):
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
    return engine


@torch.no_grad()
def test_pytorch(images, use_fp16=False):
    model = uniformer_small(flat_fc=False)
    state_dict = torch.load('./uniformer_small_in1k.pth')
    model.load_state_dict(state_dict['model'])
    model.to('cuda')
    model.eval()

    imgs = torch.Tensor(images).cuda()
    # warmup
    for _ in range(30):
        output = model(imgs)
    start = time_ns()
    for _ in range(30):
        output = model(imgs)
    stop = time_ns()
    time_per_run = (stop - start) / (30 * 1000 * 1000)

    return output, time_per_run


def test_tensorrt(engine, images, labels):
    context = engine.create_execution_context()
    context.set_binding_shape(0, images.shape)
    nb_input = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nb_output = engine.num_bindings - nb_input
    host_buf = [images.reshape(-1).astype(np.float32)]
    for i in range(nb_input, nb_input + nb_output):
        host_buf.append(np.ones(context.get_binding_shape(i),
                                dtype=trt.nptype(engine.get_binding_dtype(i))))

    dev_buf = []
    for i in range(nb_input + nb_output):
        dev_buf.append(cudart.cudaMalloc(host_buf[i].nbytes)[1])

    cudart.cudaMemcpy(dev_buf[0], host_buf[0].ctypes.data, host_buf[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # warmup
    for _ in range(10):
        context.execute_v2(dev_buf)

    start = time_ns()
    for _ in range(30):
        context.execute_v2(dev_buf)
    stop = time_ns()
    time_per_run = (stop - start) / (30 * 1000 * 1000)

    cudart.cudaMemcpy(host_buf[-1].ctypes.data, dev_buf[-1], host_buf[-1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for dbuf in dev_buf:
        cudart.cudaFree(dbuf)

    return host_buf[-1], time_per_run


if __name__ == '__main__':
    engine = load_engine(plan_file)
    for i in (1, 2, 3, 4, 8, 16, 32):
        ioFile = os.path.join(data_file, f"bs{i}.npz")
        ioData = np.load(ioFile)
        images = ioData['image']
        labels = ioData['label']

        torch_output, torch_time = test_pytorch(images)
        trt_output, trt_time = test_tensorrt(engine, images, labels)
        speedup = (torch_time - trt_time) / torch_time
        print(f"batch size: {images.shape[0]:2d}, ", end='')
        print(f"torch per run: {torch_time:6.3f}, trt per run: {trt_time:6.3f}, ", end='')

        o1 = torch_output.cpu()
        o2 = torch.from_numpy(trt_output)
        loss = torch.nn.functional.cross_entropy(o1, o2).detach().cpu().numpy()

        if np.sum(loss) < 1e-5:
            print(f"speedup: {speedup:6.3f}, output check: PASSED")
        else:
            print(f"speedup: {speedup:6.3f}, output check: FAILED")

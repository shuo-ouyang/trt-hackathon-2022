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
from uniformer.models import uniformer_small
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--engine', type=str, default='./uf.plan', help='trt engine file want to be tested')
parser.add_argument('--data', type=str, default='./data/', help='test data path')
args = parser.parse_args()


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')

ctypes.cdll.LoadLibrary('../plugins/LayerNormPlugin/LayerNorm.so')
ctypes.cdll.LoadLibrary('../plugins/GeluPlugin/Gelu.so')


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
    model.half()
    model.eval()

    imgs = torch.Tensor(images).cuda().half()
    # warmup
    for _ in range(30):
        output = model(imgs)
    start = time_ns()
    # with torch.cuda.amp.autocast(enabled=True):
    for _ in range(30):
        output = model(imgs)
    stop = time_ns()
    time_per_run = (stop - start) / (30 * 1000 * 1000)

    return output, time_per_run


def test_tensorrt(engine, images, labels):
    context = engine.create_execution_context()
    input_shape = context.get_binding_shape(0)
    if input_shape[0] != -1 and input_shape[0] != images.shape[0]:
        print(
            f"[WANRING] engine batch({input_shape[0]}) is not dynamic and mismatch with inpu({images.shape[0]}), skip this round!!!")
        return None, None
    nb_profile = engine.num_optimization_profiles
    nb_input = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nb_output = engine.num_bindings - nb_input
    host_buf = [images.reshape(-1).astype(np.float32)]
    nb_input //= nb_profile
    nb_output //= nb_profile

    if nb_profile == 1:
        bind_offset = 0
    else:
        if images.shape[0] <= 4:
            bind_offset = 0
            context.set_optimization_profile_async(0, 0)
            cudart.cudaStreamSynchronize(0)
        else:
            bind_offset = nb_profile
            context.set_optimization_profile_async(1, 0)
            cudart.cudaStreamSynchronize(0)

    context.set_binding_shape(bind_offset, images.shape)

    for i in range(nb_input, nb_input + nb_output):
        i += bind_offset
        host_buf.append(np.ones(context.get_binding_shape(i),
                                dtype=trt.nptype(engine.get_binding_dtype(i))))

    dev_buf = []
    for i in range(nb_input + nb_output):
        dev_buf.append(cudart.cudaMalloc(host_buf[i].nbytes)[1])

    if nb_profile == 1 or images.shape[0] <= 4:
        dev_buf = dev_buf + [int(0), int(0)]
    else:
        dev_buf = [int(0), int(0)] + dev_buf

    cudart.cudaMemcpy(dev_buf[bind_offset], host_buf[0].ctypes.data, host_buf[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # warmup
    for _ in range(10):
        context.execute_v2(dev_buf)

    start = time_ns()
    for _ in range(30):
        context.execute_v2(dev_buf)
    stop = time_ns()
    time_per_run = (stop - start) / (30 * 1000 * 1000)

    cudart.cudaMemcpy(host_buf[-1].ctypes.data, dev_buf[bind_offset + 1], host_buf[-1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for dbuf in dev_buf:
        cudart.cudaFree(dbuf)

    return host_buf[-1], time_per_run


if __name__ == '__main__':
    engine = load_engine(args.engine)
    for i in (1, 2, 3, 4, 8, 16, 32):
        ioFile = os.path.join(args.data, f"bs{i}.npz")
        ioData = np.load(ioFile)
        images = ioData['image']
        labels = ioData['label']

        torch_output, torch_time = test_pytorch(images)
        trt_output, trt_time = test_tensorrt(engine, images, labels)
        if trt_output is None:
            continue
        speedup = (torch_time - trt_time) / torch_time
        print(f"batch size: {images.shape[0]:2d}, ", end='')
        print(f"torch per run: {torch_time:6.3f}, trt per run: {trt_time:6.3f}, ", end='')

        o1 = torch_output.to(torch.float32).cpu()
        o2 = torch.from_numpy(trt_output.reshape(-1, 1000))
        loss = torch.nn.functional.cross_entropy(o1, o2).detach().cpu().numpy()

        if np.sum(loss) < 1e-6:
            print(f"speedup: {speedup:6.3f}, output check: PASSED")
        else:
            print(f"speedup: {speedup:6.3f}, output check: FAILED ")

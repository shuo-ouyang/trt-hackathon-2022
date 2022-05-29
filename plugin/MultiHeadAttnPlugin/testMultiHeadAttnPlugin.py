import os
import ctypes
import torch
import numpy as np
from time import time_ns
import tensorrt as trt
from cuda import cudart
from torch import nn
torch.backends.cudnn.deterministic = True

soFilePath = "./MultiHeadAttn.so"
nTime = 30

nIn, cIn, hIn, wIn = 2, 3, 4, 5
npDataType = np.float32
globalEpsilon = 1e-5
np.random.seed(97)
head_size = 2


def check(a, b, weak=False):
    print(a[:, :, :16])
    print(b[:, :, :16])
    if weak:
        return np.all(np.abs(a - b) < globalEpsilon)
    else:
        return np.all(a == b)


def mha_pytorch(bufferH, epsilon):
    data = bufferH[0].reshape(1, 8, 3*32)
    print("========================="*3)
    q = torch.Tensor(data[:, :, :32]).float().cuda()
    k = torch.Tensor(data[:, :, 32:64]).float().cuda()
    v = torch.Tensor(data[:, :, 64:]).float().cuda()

    mha = torch.nn.MultiheadAttention(32, head_size, device=0, bias=False)
    output, weight = mha(q, k, v,  average_attn_weights=False)
    return [output.cpu().detach().numpy().reshape((8, 1, 32)), weight.cpu().detach().numpy()]


def getMHAPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'MultiHeadAttn':
            type_id = trt.PluginField('type_id', np.int32(0), trt.PluginFieldType.INT32)
            hidden_size = trt.PluginField('hidden_size', np.int32(32), trt.PluginFieldType.INT32)
            num_heads = trt.PluginField('num_heads', np.int32(2), trt.PluginFieldType.INT32)
            has_mask = trt.PluginField('has_mask', np.int32(0), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([type_id, hidden_size, num_heads, has_mask]))
    return None


if __name__ == '__main__':
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    testCase = "fp%s" % ('16' if int(npDataType == np.float16) else '32')
    print("Test <%s>" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = "./model-" + testCase + ".plan"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << 0)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0

        inputTensorList = []
        trtDataType = trt.float16 if int(npDataType == np.float16) else trt.float32
        inputTensorList.append(network.add_input('inputT', trtDataType, [-1, -1, -1, 1, 1]))

        profile = builder.create_optimization_profile()
        profile.set_shape('inputT', [8, 1,  3*32, 1, 1], [8, 1, 3 * 32, 1, 1], [8, 1, 3 * 32, 1, 1])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2(inputTensorList, getMHAPlugin())
        pluginLayer.get_output(0).dtype = trtDataType

        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,  [8, 1, 3 * 32, 1, 1])

    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    _, stream = cudart.cudaStreamCreate()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i),
              engine.get_binding_shape(i), context.get_binding_shape(i))

    # data = np.arange(3*32*8).reshape(8, 1, 32*3, 1, 1).astype(npDataType)
    data = np.ones((8, 1, 32*3, 1, 1)).astype(npDataType)

    bufferH = []
    bufferH.append(data)
    bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data,
                               bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    context.execute_async_v2(bufferD, stream)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    resCPU = mha_pytorch(bufferH, globalEpsilon)
    print("check result:", check(resCPU[0], bufferH[-1].reshape(8, 1, 32), True))
    print(resCPU[1])
    print("Test <%s> finish!" % testCase)

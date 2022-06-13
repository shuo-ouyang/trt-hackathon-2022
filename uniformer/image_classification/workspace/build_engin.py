import os
import numpy as np
# from cuda import cudart
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes


onnx_file = './uf_opt.onnx'
trt_file = './uf.plan'
so_files = ['../../../plugin/LayerNormPlugin/LayerNorm.so']

os.system("rm -rf  *.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
# cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')
for so in so_files:
    ctypes.cdll.LoadLibrary(so)
if os.path.isfile(trt_file):
    with open(trt_file, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    calib_profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) |
                              1 << int(trt.TacticSource.CUBLAS_LT) |
                              1 << int(trt.TacticSource.CUDNN))
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.clear_flag(trt.BuilderFlag.TF32)
    # config.set_flag(trt.BuilderFlag.INT8)
    # config.int8_calibrator = EncoderEntropyCalibrator(calib_data_path, cache_file, 1)
    config.max_workspace_size = 24 << 30
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_file):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing ONNX file!")

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if str(layer.name).startswith('LayerNorm'):
            layer.precision = trt.DataType.FLOAT

    input = network.get_input(0)

    profile.set_shape(input.name, (1, 3, 224, 224), (4, 3, 224, 224), (16, 3, 224, 224))

    # calib_profile.set_shape(input.name, (1, 3, 224, 224), (4, 3, 224, 224), (16, 3, 224, 224))

    config.add_optimization_profile(profile)
    # config.set_calibration_profile(calib_profile)

    engine_string = builder.build_serialized_network(network, config)
    if engine_string == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trt_file, 'wb') as f:
        f.write(engine_string)
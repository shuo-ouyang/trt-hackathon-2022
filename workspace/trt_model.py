import torch
import tensorrt as trt
import numpy as np
import ctypes
from calibrator import EntropyCalibrator
import os
from cuda import cudart

logger = trt.Logger(trt.Logger.WARNING)
ctypes.cdll.LoadLibrary('../plugins/LayerNormPlugin/LayerNorm.so')
ctypes.cdll.LoadLibrary('../plugins/GeluPlugin/Gelu.so')
layer_scale = False
cache_file = './static_bs16_int8.cache'
calib_data_path = './calibration'
C, H, W = 3, 224, 224
os.system('rm ./static_bs16_int8.cache')
use_fp16 = True
use_int8 = False


def get_layernorm_plugin():
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == 'LayerNorm':
            return creator.create_plugin(creator.name, trt.PluginFieldCollection([]))
    return None


def get_gelu_plugin():
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == 'Gelu':
            return creator.create_plugin(creator.name, trt.PluginFieldCollection([]))
    return None


def find_params(params, id, sub_id=None):
    pm = {}
    for k, v in params.items():
        k = str(k)
        spk = k.split('.')
        if spk[0] in ('norm', 'head'):
            continue
        # patch embed
        if sub_id is None:
            if int(spk[0][-1]) == id and spk[0][0] == 'p':
                pm['.'.join(spk[1:])] = v
        # blocks
        else:
            if int(spk[0][-1]) == id and spk[0][0] == 'b':
                if int(spk[1]) == sub_id:
                    pm['.'.join(spk[2:])] = v
    return pm


def mlp(network, data, params, in_features, hidden_features=None, out_features=None):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    b, n, c = data.shape
    shfl1 = network.add_shuffle(data)
    shfl1.reshape_dims = (b, n, c, 1, 1)

    data = shfl1.get_output(0)
    w_fc1 = trt.Weights(params.get('fc1.weight', None))
    b_fc1 = trt.Weights(params.get('fc1.bias', None))
    fc1 = network.add_fully_connected(data, hidden_features, w_fc1, b_fc1)

    data = fc1.get_output(0)
    gelu = network.add_plugin_v2([data], get_gelu_plugin())

    data = gelu.get_output(0)
    w_fc2 = trt.Weights(params.get('fc2.weight', None))
    b_fc2 = trt.Weights(params.get('fc2.bias', None))
    fc2 = network.add_fully_connected(data, out_features, w_fc2, b_fc2)

    data = fc2.get_output(0)
    shfl2 = network.add_shuffle(data)
    shfl2.reshape_dims = (b, n, c)

    data = shfl2.get_output(0)
    return network, data


def conv_mlp(network, data, params, in_features, hidden_features=None, out_features=None):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    w_fc1 = trt.Weights(params.get('fc1.weight', None))
    b_fc1 = trt.Weights(params.get('fc1.bias', None))
    fc1 = network.add_convolution_nd(data, hidden_features, (1, 1), w_fc1, b_fc1)

    data = fc1.get_output(0)
    gelu = network.add_plugin_v2([data], get_gelu_plugin())

    data = gelu.get_output(0)
    w_fc2 = trt.Weights(params.get('fc2.weight', None))
    b_fc2 = trt.Weights(params.get('fc2.bias', None))
    fc2 = network.add_convolution_nd(data, out_features, (1, 1), w_fc2, b_fc2)

    data = fc2.get_output(0)
    return network, data


def attn(network, data, params, dim, num_heads=8):
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    b, n, c = data.shape

    input_shfl = network.add_shuffle(data)
    input_shfl.reshape_dims = (b, n, c, 1, 1)

    data = input_shfl.get_output(0)
    w_qkv = trt.Weights(params.get('qkv.weight', None))
    qkv_fc = network.add_fully_connected(data, 3 * dim, w_qkv)

    data = qkv_fc.get_output(0)
    qkv_shfl1 = network.add_shuffle(data)
    qkv_shfl1.reshape_dims = (b, n, 3, num_heads, c // num_heads)
    qkv_shfl1.second_transpose = trt.Permutation((2, 0, 3, 1, 4))

    data = qkv_shfl1.get_output(0)

    q_slice = network.add_slice(data, (0, 0, 0, 0, 0),
                                (1, b, num_heads, n, c // num_heads),
                                (1, 1, 1, 1, 1))
    q = q_slice.get_output(0)
    q_shfl = network.add_shuffle(q)
    q_shfl.reshape_dims = (b, num_heads, n, c // num_heads)
    q = q_shfl.get_output(0)

    k_slice = network.add_slice(data, (1, 0, 0, 0, 0),
                                (1, b, num_heads, n, c // num_heads),
                                (1, 1, 1, 1, 1))

    k = k_slice.get_output(0)
    k_shfl = network.add_shuffle(k)
    k_shfl.reshape_dims = (b, num_heads, n, c // num_heads)
    k = k_shfl.get_output(0)

    v_slice = network.add_slice(data, (2, 0, 0, 0, 0),
                                (1, b, num_heads, n, c // num_heads),
                                (1, 1, 1, 1, 1))
    v = v_slice.get_output(0)
    v_shfl = network.add_shuffle(v)
    v_shfl.reshape_dims = (b, num_heads, n, c // num_heads)
    v = v_shfl.get_output(0)

    qk = network.add_matrix_multiply(q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE)

    data = qk.get_output(0)
    scale = network.add_constant((1, 1, 1, 1), trt.Weights(
        np.array(scale, dtype=np.float32))).get_output(0)
    qk_scale = network.add_elementwise(data, scale, trt.ElementWiseOperation.PROD)

    data = qk_scale.get_output(0)
    softmax = network.add_softmax(data)
    softmax.axes = (1 << 3)

    data = softmax.get_output(0)
    attn = network.add_matrix_multiply(
        data, trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE)

    data = attn.get_output(0)

    attn_shfl = network.add_shuffle(data)
    attn_shfl.first_transpose = trt.Permutation((0, 2, 1, 3))
    attn_shfl.reshape_dims = (b, n, c, 1, 1)

    data = attn_shfl.get_output(0)
    w_attn = trt.Weights(params.get('proj.weight', None))
    b_attn = trt.Weights(params.get('proj.bias', None))
    attn_fc = network.add_fully_connected(data, dim, w_attn, b_attn)
    data = attn_fc.get_output(0)

    output_shfl = network.add_shuffle(data)
    output_shfl.reshape_dims = (b, n, c)
    data = output_shfl.get_output(0)
    return network, data


def patch_embed(network, data, params, embed_dim, patch_size):
    w_proj = trt.Weights(params.get('proj.weight', None))
    b_proj = trt.Weights(params.get('proj.bias', None))
    proj = network.add_convolution_nd(data, embed_dim, (patch_size, patch_size), w_proj, b_proj)
    proj.stride = (patch_size, patch_size)

    data = proj.get_output(0)
    b, c, h, w = data.shape
    shfl1 = network.add_shuffle(data)
    shfl1.first_transpose = trt.Permutation((0, 2, 3, 1))
    shfl1.reshape_dims = (-1, embed_dim)

    data = shfl1.get_output(0)
    w_ln = params.get('norm.weight', None)
    b_ln = params.get('norm.bias', None)
    w_ln = network.add_constant(w_ln.shape, trt.Weights(w_ln)).get_output(0)
    b_ln = network.add_constant(b_ln.shape, trt.Weights(b_ln)).get_output(0)
    ln = network.add_plugin_v2([data, w_ln, b_ln], get_layernorm_plugin())

    data = ln.get_output(0)
    shfl2 = network.add_shuffle(data)
    shfl2.reshape_dims = (b, h, w, embed_dim)
    shfl2.second_transpose = trt.Permutation((0, 3, 1, 2))

    data = shfl2.get_output(0)
    return network, data


def conv_block(network, data, params, dim, mlp_ratio=4):

    w_pos_embed = trt.Weights(params.get('pos_embed.weight', None))
    b_pos_embed = trt.Weights(params.get('pos_embed.bias', None))
    pos_embed = network.add_convolution_nd(data, dim, (3, 3), w_pos_embed, b_pos_embed)
    pos_embed.padding = (1, 1)
    pos_embed.num_groups = dim

    # data = pos_embed.get_output(0)

    add1 = network.add_elementwise(pos_embed.get_output(0), data,
                                   trt.ElementWiseOperation.SUM)

    data = add1.get_output(0)
    gamma = params.get('norm1.weight')
    beta = params.get('norm1.bias')
    mean = params.get('norm1.running_mean')
    var = params.get('norm1.running_var')
    std = np.sqrt(var + 1e-5)
    scale = trt.Weights(gamma / std)
    shift = trt.Weights((-mean * gamma) / std + beta)
    norm1 = network.add_scale(data, trt.ScaleMode.CHANNEL, shift, scale)

    w_conv1 = trt.Weights(params.get('conv1.weight'))
    b_conv1 = trt.Weights(params.get('conv1.bias'))
    conv1 = network.add_convolution_nd(norm1.get_output(0), dim, (1, 1), w_conv1, b_conv1)

    w_attn = trt.Weights(params.get('attn.weight'))
    b_attn = trt.Weights(params.get('attn.bias'))
    attn = network.add_convolution_nd(conv1.get_output(0), dim, (5, 5), w_attn, b_attn)
    attn.padding = (2, 2)
    attn.num_groups = dim

    w_conv2 = trt.Weights(params.get('conv2.weight'))
    b_conv2 = trt.Weights(params.get('conv2.bias'))
    conv2 = network.add_convolution_nd(attn.get_output(0), dim, (1, 1), w_conv2, b_conv2)

    add2 = network.add_elementwise(conv2.get_output(0), data,
                                   trt.ElementWiseOperation.SUM)

    data = add2.get_output(0)
    gamma = params.get('norm2.weight')
    beta = params.get('norm2.bias')
    mean = params.get('norm2.running_mean')
    var = params.get('norm2.running_var')
    std = np.sqrt(var + 1e-5)
    scale = trt.Weights(gamma / std)
    shift = trt.Weights((-mean * gamma) / std + beta)
    norm2 = network.add_scale(data, trt.ScaleMode.CHANNEL, shift, scale)

    mlp_params = {
        'fc1.weight': params.get('mlp.fc1.weight'),
        'fc1.bias': params.get('mlp.fc1.bias'),
        'fc2.weight': params.get('mlp.fc2.weight'),
        'fc2.bias': params.get('mlp.fc2.bias')
    }
    mlp_hidden_dim = mlp_ratio * dim
    network, mlp_out = conv_mlp(network, norm2.get_output(0), mlp_params, dim, mlp_hidden_dim)
    add3 = network.add_elementwise(mlp_out, data,
                                   trt.ElementWiseOperation.SUM)

    data = add3.get_output(0)
    return network, data


def attn_block(network, data, params, dim, num_heads, mlp_ratio=4):

    w_pos_embed = trt.Weights(params.get('pos_embed.weight', None))
    b_pos_embed = trt.Weights(params.get('pos_embed.bias', None))
    pos_embed = network.add_convolution_nd(data, dim, (3, 3), w_pos_embed, b_pos_embed)
    pos_embed.padding = (1, 1)
    pos_embed.num_groups = dim
    add1 = network.add_elementwise(pos_embed.get_output(0), data,
                                   trt.ElementWiseOperation.SUM)

    data = add1.get_output(0)
    b, n, h, w = data.shape
    shfl1 = network.add_shuffle(data)
    shfl1.reshape_dims = (b, n, h * w)
    shfl1.second_transpose = trt.Permutation((0, 2, 1))

    data = shfl1.get_output(0)
    gamma = params.get('norm1.weight')
    beta = params.get('norm1.bias')
    gamma = network.add_constant(gamma.shape, trt.Weights(gamma)).get_output(0)
    beta = network.add_constant(beta.shape, trt.Weights(beta)).get_output(0)
    norm1 = network.add_plugin_v2([data, gamma, beta], get_layernorm_plugin())

    attn_params = {
        'qkv.weight': params.get('attn.qkv.weight'),
        'proj.weight': params.get('attn.proj.weight'),
        'proj.bias': params.get('attn.proj.bias')
    }
    network, attn_out = attn(network, norm1.get_output(0), attn_params, dim, num_heads)
    add2 = network.add_elementwise(attn_out, data,
                                   trt.ElementWiseOperation.SUM)

    data = add2.get_output(0)
    gamma = params.get('norm2.weight')
    beta = params.get('norm2.bias')
    gamma = network.add_constant(gamma.shape, trt.Weights(gamma)).get_output(0)
    beta = network.add_constant(beta.shape, trt.Weights(beta)).get_output(0)
    norm2 = network.add_plugin_v2([data, gamma, beta], get_layernorm_plugin())

    mlp_hidden_dim = int(mlp_ratio * dim)
    mlp_params = {
        'fc1.weight': params.get('mlp.fc1.weight'),
        'fc1.bias': params.get('mlp.fc1.bias'),
        'fc2.weight': params.get('mlp.fc2.weight'),
        'fc2.bias': params.get('mlp.fc2.bias')
    }
    network, mlp_out = mlp(network, norm2.get_output(0), mlp_params, dim, mlp_hidden_dim)

    add3 = network.add_elementwise(mlp_out, data,
                                   trt.ElementWiseOperation.SUM)

    data = add3.get_output(0)
    shfl2 = network.add_shuffle(data)
    shfl2.first_transpose = trt.Permutation((0, 2, 1))
    shfl2.reshape_dims = (b, n, h, w)

    data = shfl2.get_output(0)
    return network, data


def uniformer(network, data, params):
    # patch_embed_1
    pms = find_params(params, 1)
    network, data = patch_embed(network, data, pms, 64, 4)

    # blocks_1
    pms = find_params(params, 1, 0)
    network, data = conv_block(network, data, pms, 64)
    pms = find_params(params, 1, 1)
    network, data = conv_block(network, data, pms, 64)
    pms = find_params(params, 1, 2)
    network, data = conv_block(network, data, pms, 64)

    # patch_embed_2
    pms = find_params(params, 2)
    network, data = patch_embed(network, data, pms, 128, 2)

    # blocks_2
    pms = find_params(params, 2, 0)
    network, data = conv_block(network, data, pms, 128)
    pms = find_params(params, 2, 1)
    network, data = conv_block(network, data, pms, 128)
    pms = find_params(params, 2, 2)
    network, data = conv_block(network, data, pms, 128)
    pms = find_params(params, 2, 3)
    network, data = conv_block(network, data, pms, 128)

    # patch_embed_3
    pms = find_params(params, 3)
    network, data = patch_embed(network, data, pms, 320, 2)

    # blocks_3
    pms = find_params(params, 3, 0)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 1)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 2)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 3)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 4)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 5)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 6)
    network, data = attn_block(network, data, pms, 320, 5)
    pms = find_params(params, 3, 7)
    network, data = attn_block(network, data, pms, 320, 5)

    # patch_embed_4
    pms = find_params(params, 4)
    network, data = patch_embed(network, data, pms, 512, 2)

    # blocks_4
    pms = find_params(params, 4, 0)
    network, data = attn_block(network, data, pms, 512, 8)
    pms = find_params(params, 4, 1)
    network, data = attn_block(network, data, pms, 512, 8)
    pms = find_params(params, 4, 2)
    network, data = attn_block(network, data, pms, 512, 8)

    # norm
    gamma = params.get('norm.weight')
    beta = params.get('norm.bias')
    mean = params.get('norm.running_mean')
    var = params.get('norm.running_var')
    std = np.sqrt(var + 1e-5)
    scale = trt.Weights(gamma / std)
    shift = trt.Weights((-mean * gamma) / std + beta)
    norm = network.add_scale(data, trt.ScaleMode.CHANNEL, shift, scale)

    data = norm.get_output(0)
    b, n, h, w = data.shape
    shfl1 = network.add_shuffle(data)
    shfl1.reshape_dims = (b, n, h * w)

    data = shfl1.get_output(0)

    reduce_mean = network.add_reduce(data, trt.ReduceOperation.AVG, 4, False)

    # shuffle for last fc
    data = reduce_mean.get_output(0)
    shfl2 = network.add_shuffle(data)
    shfl2.reshape_dims = (b, 512, 1, 1)

    # classification head
    data = shfl2.get_output(0)
    w_head = trt.Weights(params.get('head.weight'))
    b_head = trt.Weights(params.get('head.bias'))
    head = network.add_fully_connected(data, 1000, w_head, b_head)
    head.precision = trt.float32
    head.get_output(0).dtype = trt.float32
    data = head.get_output(0)
    return network, data


if __name__ == '__main__':
    model_path = './uniformer_small_in1k.pth'
    state_dict = torch.load(model_path)
    params = {k: v.detach().cpu().numpy() for k, v in state_dict['model'].items()}

    for B in (1, 2, 3, 4, 8, 16, 32):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        input = network.add_input('input', trt.float32, [B, C, H, W])
        network, output = uniformer(network, input, params)
        network.mark_output(output)

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        typstr = 'fp32'
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            typstr = 'fp16'
        if use_int8:
            if B != 16:
                print("[WARNING] model built from trt api only support bs=16 with int8 mode!!!")
                continue
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = EntropyCalibrator(calib_data_path, cache_file.format(B), 32)
            calib_profile = builder.create_optimization_profile()
            calib_profile.set_shape(input.name, (B, 3, 224, 224), (B, 3, 224, 224), (B, 3, 224, 224))
            config.set_calibration_profile(calib_profile)
            typstr = 'int8'
        config.max_workspace_size = 12 << 30

        profile = builder.create_optimization_profile()
        profile.set_shape(input.name, (B, 3, 224, 224), (B, 3, 224, 224), (B, 3, 224, 224))
        config.add_optimization_profile(profile)

        engine_str = builder.build_serialized_network(network, config)
        with open(f'./uf_static_bs{B}_{typstr}.plan', 'wb') as f:
            f.write(engine_str)
        cudart.cudaDeviceSynchronize()

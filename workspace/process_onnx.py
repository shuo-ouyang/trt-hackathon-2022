import onnx_graphsurgeon as gs
import onnx
import numpy as np
from polygraphy.backend.onnx import FoldConstants
import argparse

use_fp16 = True
parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, default='./uf_small.onnx', help='onnx file path')
parser.add_argument('--use-fp16', action='store_true', help='whether use fp16 specific modification')
parser.add_argument('--use-ln', action='store_true', help='whether use layer norm plugin')
parser.add_argument('--use-gelu', action='store_true', help='whether use gelu plugin')
parser.add_argument('--output', type=str, default='./uf_opt.onnx', help='output optimized onnx file')

args = parser.parse_args()

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(args.onnx)))
tmap = graph.tensors()

# solve conv group issue
if args.use_fp16:
    for node in graph.nodes:
        if node.name in ('Reshape_36', 'Reshape_149', 'Reshape_287', 'Reshape_1357'):
            node.i(1).inputs[-1].values = np.array([node.i().inputs[-1].shape[0]])
else:
    for node in graph.nodes:
        if node.name in ('Reshape_34', 'Reshape_127', 'Reshape_239', 'Reshape_1243'):
            node.i(1).inputs[-1].values = np.array([node.i().inputs[-1].shape[0]])

# layer norm
if args.use_ln:
    reduce_mean_nodes = [node for node in graph.nodes if node.op == 'ReduceMean' and node.o().op == 'Sub']
    for idx, node in enumerate(reduce_mean_nodes):

        gamma = node.o().o(1).o().inputs[1]
        beta = node.o().o(1).o().o().inputs[1]

        inputs = [node.inputs[0], gamma, beta]
        outputs = node.o().o(0).o().o().o().o().o().o().outputs
        ln_node = gs.Node(op='LayerNorm', name='LayerNorm_{}'.format(idx), inputs=inputs, outputs=outputs)
        graph.nodes.append(ln_node)

        node.o().o().o().o().o().inputs.clear()
        node.o().o(1).o().o().outputs.clear()

# gelu
if args.use_gelu:
    erf_nodes = [node for node in graph.nodes if node.op == 'Erf']
    for idx, node in enumerate(erf_nodes):
        inputs = [node.i().inputs[0]]
        outputs = [node.o().o().o().outputs[0]]
        gelu_node = gs.Node(op='Gelu', name='Gelu_{}'.format(idx), inputs=inputs, outputs=outputs)
        graph.nodes.append(gelu_node)

        node.i().inputs.clear()
        node.o().o().o().outputs.clear()


graph.cleanup().toposort()
opt_graph = FoldConstants(gs.export_onnx(graph))()
onnx.save(opt_graph, './uf_opt.onnx')

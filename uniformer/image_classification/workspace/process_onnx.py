import onnx_graphsurgeon as gs
import onnx
import numpy as np
from polygraphy.backend.onnx import FoldConstants

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load('./uf_small.onnx')))
tmap = graph.tensors()

tmap['1893'].values = np.array([64])
tmap['1894'].values = np.array([128])
tmap['1895'].values = np.array([320])
tmap['1944'].values = np.array([512])

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

graph.cleanup().toposort()
opt_graph = FoldConstants(gs.export_onnx(graph))()
onnx.save(opt_graph, './uf_opt.onnx')

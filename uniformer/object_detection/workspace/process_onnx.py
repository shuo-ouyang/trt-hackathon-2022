import onnx_graphsurgeon as gs
import onnx
import numpy as np
from polygraphy.backend.onnx import FoldConstants

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load('./uf_small.onnx')))
tmap = graph.tensors()

tmap['5171'].values = np.array([64])
tmap['5172'].values = np.array([128])
tmap['5173'].values = np.array([320])

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

# reshape_nodes = [node for node in graph.nodes if node.op == 'Reshape']
# for node in reshape_nodes:
#     if isinstance(node.inputs[1].shape[0], int) and node.inputs[1].shape[0] == 2:
#         if 'value' in node.inputs[1].inputs[0].attrs.keys():
#             shape = node.inputs[1].inputs[0].attrs['value'].values
#             if len(shape) >=2:
#                 print(node.name)

constant_0 = gs.Constant(name='Constant_0', values=np.ascontiguousarray(np.array([0], dtype=np.int64)))
constant_n1 = gs.Constant(name='Constant_n1', values=np.ascontiguousarray(np.array([-1], dtype=np.int64)))
constant_n2 = gs.Constant(name='Constant_n2', values=np.ascontiguousarray(np.array([-2], dtype=np.int64)))
constant_p1 = gs.Constant(name='Constant_p1', values=np.ascontiguousarray(np.array([1], dtype=np.int64)))
constant_p2 = gs.Constant(name='Constant_p2', values=np.ascontiguousarray(np.array([2], dtype=np.int64)))
constant_p3 = gs.Constant(name='Constant_p3', values=np.ascontiguousarray(np.array([3], dtype=np.int64)))
constant_nmax = gs.Constant(name='Constant_nmax', values=np.ascontiguousarray(
    np.array([-9223372036854775807], dtype=np.int64)))

reshape_node = [node for node in graph.nodes if node.name == 'Reshape_331'][0]

slice_output0 = gs.Variable(name='slice_output0', dtype=np.int64, shape=[4])
slice_output1 = gs.Variable(name='slice_output1', dtype=np.int64, shape=[4])
inputs0 = [reshape_node.i(0).outputs[0], constant_n1, constant_nmax, constant_0, constant_n2]
inputs1 = [reshape_node.i(0).outputs[0], constant_n2, constant_nmax, constant_0, constant_n2]

slice_node0 = gs.Node(op='Slice', name='Slice_0', inputs=inputs0, outputs=[slice_output0])
slice_node1 = gs.Node(op='Slice', name='Slice_1', inputs=inputs1, outputs=[slice_output1])

concat_output = gs.Variable(name='concat_output0', dtype=np.int64, shape=[8])
concat_node = gs.Node(op='Concat', name='Concat_0', inputs=[slice_output0, slice_output1], outputs=[concat_output], attrs={'axis': 0})

reshape_node.o().o().o().o().inputs = concat_node.outputs
reshape_node.inputs.clear()

graph.nodes.append(slice_node0)
graph.nodes.append(slice_node1)
graph.nodes.append(concat_node)

graph.cleanup().toposort()
opt_graph = FoldConstants(gs.export_onnx(graph))()
onnx.save(opt_graph, './uf_opt.onnx')

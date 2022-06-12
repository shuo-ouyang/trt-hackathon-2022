import onnx_graphsurgeon as gs
import onnx
import numpy as np
from polygraphy.backend.onnx import FoldConstants

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load('./uf_small.onnx')))

reshape_nodes = [node for node in graph.nodes if node.op == 'Reshape']
for node in reshape_nodes:
    if isinstance(node.inputs[1].shape[0], int) and node.inputs[1].shape[0] == 2:
        if 'value' in node.inputs[1].inputs[0].attrs.keys():
            shape = node.inputs[1].inputs[0].attrs['value'].values
            node.inputs[1].inputs[0].attrs['value'].values = np.array([shape[0], shape[1], 1])
            print(node)

graph.cleanup().toposort()
opt_graph = FoldConstants(gs.export_onnx(graph))()
onnx.save(opt_graph, './uf_opt.onnx')

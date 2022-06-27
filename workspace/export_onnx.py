
import torch
import torch.onnx

import sys
sys.path.append('../')
from uniformer.models import uniformer_small, uniformer_base

device = 'cuda'
model_path = './uniformer_small_in1k.pth'
model = uniformer_small(flat_fc=False)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model'])

model = model.to(device)
model.eval()
# model.half()
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

with torch.cuda.amp.autocast():
    torch.onnx.export(model,
                      dummy_input,
                      'uf_small.onnx',
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'output': {0: 'batch_size'}
                      },
                      export_params=True)

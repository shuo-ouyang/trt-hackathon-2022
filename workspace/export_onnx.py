
import torch
import torch.onnx
import argparse
import sys
sys.path.append('../')
from uniformer.models import uniformer_small
parser = argparse.ArgumentParser()

parser.add_argument('--use-fp16', action='store_true', help='whether use fp16')

device = 'cuda'
model_path = './uniformer_small_in1k.pth'
model = uniformer_small(flat_fc=False)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model'])

model = model.to(device)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
args = parser.parse_args()
with torch.cuda.amp.autocast(enabled=args.use_fp16):
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

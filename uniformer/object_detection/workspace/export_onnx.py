import os
import numpy as np
import torch
import torch.nn.functional as F
# import torchvision.transforms as T
import torch.onnx

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import mmcv
from mmcv.parallel import collate, scatter

device = 'cpu'
ckpt_file = './mask_rcnn_3x_ms_hybrid_small.pth'
cfg_file = './exp/mask_rcnn_3x_ms_hybrid_small/config.py'

model = init_detector(cfg_file, ckpt_file, device=device)

res, data = inference_detector(model, '/mnt/e/code-repos/UniFormer/object_detection/demo/demo.jpg')

img = data['img']
img_meta = data['img_metas'][0]
img_meta[0]['scale_factor'] = torch.from_numpy(img_meta[0]['scale_factor'])
img_meta[0]['img_norm_cfg']['mean'] = torch.from_numpy(img_meta[0]['img_norm_cfg']['mean'])
img_meta[0]['img_norm_cfg']['std'] = torch.from_numpy(img_meta[0]['img_norm_cfg']['std'])
img_metas = [img_meta]

torch.onnx.export(model,
                  (img, img_metas),
                  'uf_mask_rcnn_3x_small.onnx',
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input0', 'input1', 'input2'],
                  output_names=['output0', 'output1', 'output2'],
                  export_params=True,
                  dynamic_axes={'input0': [0],
                                'output0': [0],
                                'outpu1': [0],
                                'output2': [0]}
                  )

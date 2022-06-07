from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from imagenet_class_index import imagenet_classnames

import sys
sys.path.append('../')
from models.uniformer import uniformer_small


imagenet_id_to_classname = {}
for k, v in imagenet_classnames.items():
    imagenet_id_to_classname[k] = v[1]


def inference(model, image):
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.to('cuda').unsqueeze(0)

    pred = model(image)
    pred = F.softmax(pred, dim=1).flatten()

    # return pred.topk(5)
    return {imagenet_id_to_classname[str(i)]: float(pred[i]) for i in range(1000)}


device = 'cuda'
model_path = './uniformer_small_in1k.pth'
model = uniformer_small()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model'])

model = model.to(device)
model.eval()


image = Image.open('./val_3.JPEG')
infer = inference(model, image)

max_score = 0
pred = None
for name, score in infer.items():
    if score > max_score:
        pred = name
        max_score = score
print(pred, max_score)

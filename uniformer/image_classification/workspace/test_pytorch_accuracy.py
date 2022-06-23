
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets.folder import ImageFolder
from imagenet_class_index import imagenet_classnames
import utils
from timm.utils import accuracy
import numpy as np

import sys
sys.path.append('../')
from models.uniformer import uniformer_small


imagenet_id_to_classname = {}
for k, v in imagenet_classnames.items():
    imagenet_id_to_classname[k] = v[1]


def build_dataset(root, transform):
    dataset = ImageFolder(root, transform=transform)
    nb_classes = 1000
    return dataset, nb_classes


def build_transform():
    transform = T.Compose([
        T.Resize(224, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'PyTorch: '

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('PyTorch: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    device = 'cuda'
    model_path = './uniformer_small_in1k.pth'
    model = uniformer_small()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])

    model = model.to(device)
    model.eval()

    transform = build_transform()
    dataset, _ = build_dataset('/imagenet/val', transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(1),
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    # CustomQKVToContextPluginDynamic
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset)} test images: {test_stats['acc1']:.1f}%")

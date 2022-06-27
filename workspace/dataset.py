import torchvision.transforms as T
from torchvision.datasets.folder import ImageFolder
import utils
from timm.utils import accuracy


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

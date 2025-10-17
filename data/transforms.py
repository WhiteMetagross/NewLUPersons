#Image transformation utilities for NewLUPersons.

from typing import List, Tuple
import torchvision.transforms as transforms


#Get standard image transformation pipeline for inference.
def get_inference_transforms(
    image_size: int = 256,
    crop_size: int = 224,
    mean: List[float] = None,
    std: List[float] = None,
) -> transforms.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


#Get image transformation pipeline for training with augmentation.
def get_training_transforms(
    image_size: int = 256,
    crop_size: int = 224,
    mean: List[float] = None,
    std: List[float] = None,
    random_flip: bool = True,
    random_crop: bool = True,
) -> transforms.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if random_crop:
        transform_list.append(transforms.RandomCrop(crop_size))
    else:
        transform_list.append(transforms.CenterCrop(crop_size))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)

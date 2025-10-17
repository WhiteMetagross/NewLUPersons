#Default configuration for NewLUPersons models and inference.

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ModelConfig:
    #Configuration for model loading and inference.
    backbone: str = "resnet50"
    num_classes: int = 1000
    image_size: int = 256
    crop_size: int = 224
    feature_dim: int = 2048
    device: str = "cuda"
    pretrained: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    #Configuration for data loading and preprocessing.
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: bool = True
    random_flip: bool = True
    random_crop: bool = True
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class InferenceConfig:
    #Configuration for inference.
    extract_features: bool = True
    normalize_features: bool = True
    batch_size: int = 32
    return_dict: bool = True


#Get default configuration for NewLUPersons.
def get_default_config() -> Dict[str, Any]:
    return {
        "model": ModelConfig(),
        "data": DataConfig(),
        "inference": InferenceConfig(),
    }

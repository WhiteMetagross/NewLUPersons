#Helper utilities for NewLUPersons.

from typing import Optional
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image


#Get the appropriate device for computation.
def get_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


#Set random seed for reproducibility.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    #For deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Load an image from file.
def load_image(image_path: str) -> Image.Image:
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return Image.open(image_path).convert('RGB')


#Save feature vectors to file.
def save_features(
    features: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {'features': features}
    if metadata:
        data['metadata'] = metadata
    
    np.savez_compressed(output_path, **data)


#Load feature vectors from file.
def load_features(
    input_path: str,
) -> tuple:
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")
    
    data = np.load(input_path, allow_pickle=True)
    
    features = data['features']
    metadata = data.get('metadata', None)
    
    if metadata is not None:
        metadata = metadata.item()
    
    return features, metadata

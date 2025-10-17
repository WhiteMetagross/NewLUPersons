#Dataset utilities for NewLUPersons.

from typing import Optional, Callable, List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    #Simple dataset for loading images from a directory or list of paths.
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
    ):
        #Initialize ImageDataset with image paths and optional transform.
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        #Return the number of images in the dataset.
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        #Load and return an image at the given index.
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    @classmethod
    def from_directory(
        cls,
        directory: str,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        transform: Optional[Callable] = None,
    ) -> 'ImageDataset':
        #Create dataset from all images in a directory.
        directory = Path(directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(sorted(directory.glob(f'*{ext}')))
            image_paths.extend(sorted(directory.glob(f'*{ext.upper()}')))
        
        image_paths = [str(p) for p in sorted(set(image_paths))]
        
        return cls(image_paths, transform)

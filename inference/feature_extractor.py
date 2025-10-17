#Feature extraction utilities for NewLUPersons.

from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ..models import load_model
from ..data import get_inference_transforms, ImageDataset


class FeatureExtractor:
    #Extract deep features from person images using a pre-trained model.
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
    ):
        #Initialize FeatureExtractor with model path and configuration.
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        
        #Load model.
        self.model = load_model(model_path, device=device)
        self.model.eval()
        
        #Get feature dimension.
        self.feature_dim = self._get_feature_dim()
    
    def _get_feature_dim(self) -> int:
        #Infer feature dimension from model.
        if hasattr(self.model, 'fc'):
            return self.model.fc.in_features
        return 2048
    
    def extract_features(
        self,
        images: Union[str, List[str], torch.Tensor],
    ) -> np.ndarray:
        #Extract features from images (single path, list of paths, or tensor).
        if isinstance(images, str):
            images = [images]
        
        if isinstance(images, list):
            #Load images from paths.
            transform = get_inference_transforms()
            dataset = ImageDataset(images, transform=transform)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        elif isinstance(images, torch.Tensor):
            #Use tensor directly.
            if images.dim() == 3:
                images = images.unsqueeze(0)
            dataloader = [(images,)]
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")
        
        #Extract features.
        features_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                
                #Forward pass.
                with torch.no_grad():
                    #Get features from penultimate layer.
                    if hasattr(self.model, 'fc'):
                        #Remove final classification layer.
                        features = self.model(batch)
                        if isinstance(features, dict):
                            features = features.get('features', features)
                    else:
                        features = self.model(batch)
                
                #Normalize if requested.
                if self.normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)

                features_list.append(features.cpu().detach().numpy())
        
        #Concatenate all features.
        all_features = np.concatenate(features_list, axis=0)
        
        return all_features
    
    def extract_features_from_directory(
        self,
        directory: str,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ) -> Dict[str, np.ndarray]:
        #Extract features from all images in a directory.
        directory = Path(directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(sorted(directory.glob(f'*{ext}')))
            image_paths.extend(sorted(directory.glob(f'*{ext.upper()}')))
        
        image_paths = sorted(set(image_paths))
        
        if not image_paths:
            raise ValueError(f"No images found in {directory}")
        
        #Extract features.
        features = self.extract_features([str(p) for p in image_paths])
        
        #Create mapping.
        result = {}
        for path, feature in zip(image_paths, features):
            result[path.name] = feature
        
        return result

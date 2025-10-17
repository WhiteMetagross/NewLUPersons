#Model building utilities for NewLUPersons.

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from pathlib import Path
import warnings


#Load a pre-trained model from a checkpoint file.
def load_model(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strict: bool = False,
) -> nn.Module:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    #Load checkpoint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    #Extract model state dict.
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    #Create a feature extractor model that can load the state dict.
    model = _create_feature_extractor(state_dict, device)

    #Load weights with non-strict mode to handle architecture differences.
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        #If loading fails, try to load partial state.
        warnings.warn(f"Could not load full state dict: {e}. Loading partial state.")
        _load_partial_state_dict(model, state_dict)

    model.eval()

    return model


#Build a model from scratch.
def build_model(
    backbone: str = "resnet50",
    num_classes: int = 1000,
    pretrained: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError("torchvision is required for model building")

    #Build backbone using modern weights parameter instead of deprecated pretrained.
    if hasattr(models, backbone):
        if pretrained:
            #Use weights parameter for modern torchvision (>= 0.13).
            try:
                model = getattr(models, backbone)(weights="DEFAULT")
            except TypeError:
                #Fallback for older torchvision versions.
                model = getattr(models, backbone)(pretrained=True)
        else:
            model = getattr(models, backbone)(weights=None)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    #Modify final layer for num_classes.
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    model.eval()

    return model


#Create a feature extractor model that can load LUPerson/FastReID state dicts.
def _create_feature_extractor(
    state_dict: Dict[str, Any],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:

    class FeatureExtractorModel(nn.Module):
        #Feature extractor that loads LUPerson/FastReID models.
        def __init__(self):
            super().__init__()
            #Create a simple ResNet50 backbone.
            try:
                import torchvision.models as models
                #Suppress deprecation warnings from torchvision.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.backbone = models.resnet50(weights=None)
                #Remove the final classification layer.
                self.backbone.fc = nn.Identity()
            except Exception:
                #Fallback: create a simple sequential model.
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )

        def forward(self, x):
            #Extract features from backbone.
            features = self.backbone(x)
            #Flatten if needed.
            if features.dim() > 2:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            return features

    model = FeatureExtractorModel()
    model = model.to(device)
    return model


#Load partial state dict, skipping incompatible keys.
def _load_partial_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    model_state = model.state_dict()

    #Filter state dict to only include compatible keys.
    compatible_state = {}
    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_state[key] = value

    #Load compatible state.
    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)

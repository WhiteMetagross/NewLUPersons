#Inference module for NewLUPersons.

from .feature_extractor import FeatureExtractor
from .predictor import LUPersonPredictor

__all__ = [
    "FeatureExtractor",
    "LUPersonPredictor",
]

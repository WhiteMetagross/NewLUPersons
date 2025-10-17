#NewLUPersons: A modern library for person re-identification based on LUPerson and newFastReID.

__version__ = "0.1.0"
__author__ = "Mridankan Mandal"

#Core imports.
from . import config
from . import data
from . import models
from . import inference
from . import evaluation
from . import utils

#Main API.
from .inference import LUPersonPredictor, FeatureExtractor

#Get the current version of NewLUPersons.
def get_version():
    return __version__

#Public API.
__all__ = [
    "__version__",
    "get_version",
    "config",
    "data",
    "models",
    "inference",
    "evaluation",
    "utils",
    "LUPersonPredictor",
    "FeatureExtractor",
]

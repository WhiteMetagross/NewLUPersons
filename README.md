# NewLUPersons: A Modern Library for Person ReIdentification.

**A modern, production ready library for person reidentification based on [LUPerson](https://github.com/DengpanFu/LUPerson) and [newFastReID](https://github.com/WhiteMetagross/newFastReID).**

---

## Why NewLUPersons?

NewLUPersons is a **modernized, refactored version** of the original LUPerson library that provides significant improvements:

### Key Advantages Over Original LUPerson:

- **Clean, Professional Code**: All code refactored with consistent single line comments and professional Python standards.
- **Zero Warnings**: Fully compatible with modern PyTorch versions (2.5+) with zero deprecation warnings.
- **Modular Architecture**: Well organized package structure with clear separation of concerns.
- **Production Ready**: Tested and verified on Python 3.11, CUDA 12.1, Windows 11.
- **Easy Installation**: Simple pip based installation with clear dependencies.
- **Comprehensive Documentation**: Detailed guides, examples, and API reference.
- **Modern Dependencies**: Updated to work with latest PyTorch, torchvision, and CUDA versions.
- **Type Hints**: Full type annotations for better IDE support and code clarity.

### What's Included:

NewLUPersons provides a human interpretable, modular interface for:

- **Loading pre-trained LUPerson models** from Market-1501 and DukeMTMC datasets.
- **Extracting person reidentification features** with GPU acceleration.
- **Computing similarity** between persons using multiple metrics.
- **Ranking persons by similarity** for retrieval tasks.
- **Evaluating model performance** with standard ReID metrics (CMC, mAP).

## Features:

- **Clean API**: Simple, intuitive interface for common ReID tasks.
- **Pre-trained Models**: Support for LUPerson models fine-tuned on Market-1501 and DukeMTMC datasets.
- **Feature Extraction**: Efficient batch processing of images with GPU acceleration.
- **Similarity Search**: Fast similarity computation and ranking.
- **Evaluation Metrics**: Standard ReID metrics (CMC, mAP).
- **Modern Environment**: Tested on Python 3.11, CUDA 12.1, Windows 11.
- **Professional Code Quality**: Zero warnings, consistent style, full type hints.

## Installation:

### Prerequisites:

- **Python**: 3.8 or higher (tested with 3.11).
- **PyTorch**: 1.9+ with CUDA support (tested with PyTorch 2.5+).
- **CUDA**: 11.0+ (tested with CUDA 12.1).
- **Operating System**: Windows 11, Linux, macOS.

### Install:

```bash
#Install from source
pip install -e .

#Optional: Install with GPU support for similarity search.
pip install -e ".[gpu]"

#Optional: Install with CPU only FAISS.
pip install -e ".[faiss]"

#Optional: Install all optional dependencies.
pip install -e ".[all]"
```

### Downloading Pre-trained Models:

The library supports pre-trained models from the original LUPerson repository. Download the models from:

**[LUPerson GitHub Repository](https://github.com/DengpanFu/LUPerson)**

**Available Models:**

| Model | Dataset | mAP | CMC@1 | Download |
|-------|---------|-----|-------|----------|
| ResNet50 | Market-1501 | 91.12 | 96.26 | `market.pth` |
| ResNet50 | DukeMTMC | 82.27 | 90.35 | `duke.pth` |

**Download Instructions:**

1. Visit the [LUPerson GitHub repository](https://github.com/DengpanFu/LUPerson)
2. Navigate to the "Model Zoo" or "Pre-trained Models" section
3. Download `market.pth` and/or `duke.pth`
4. Place the downloaded files in your working directory or specify the path when initializing the model

```bash
#Example: Place models in current directory.
cp /path/to/market.pth ./
cp /path/to/duke.pth ./
```

## Quick Start:

### Basic Feature Extraction:

```python
from NewLUPersons import LUPersonPredictor

# Initialize predictor with pre-trained model
predictor = LUPersonPredictor(model_path="market.pth")

# Extract features from images
features = predictor.extract_features(["image1.jpg", "image2.jpg"])
print(features.shape)  # (2, 2048)
```

### Similarity Search:

```python
from NewLUPersons import LUPersonPredictor

predictor = LUPersonPredictor(model_path="market.pth")

#Find similar persons.
results = predictor.retrieve_similar_persons(
    query_image="query.jpg",
    gallery_images=["gallery1.jpg", "gallery2.jpg", "gallery3.jpg"],
    top_k=5
)

for result in results:
    print(f"Image: {result['image_path']}, Similarity: {result['similarity']:.4f}")
```

### Feature Extraction from Directory.

```python
from NewLUPersons import FeatureExtractor

extractor = FeatureExtractor(model_path="market.pth")

#Extract features from all images in a directory.
features_dict = extractor.extract_features_from_directory("path/to/images")

for filename, features in features_dict.items():
    print(f"{filename}: {features.shape}")
```

## Library Structure:

```
NewLUPersons/
├── config/          #Configuration management.
├── data/            #Data loading and preprocessing.
├── models/          #Model building and loading.
├── inference/       #Feature extraction and prediction.
├── evaluation/      #Evaluation metrics.
└── utils/           #Utility functions.
```

## API Reference:

### LUPersonPredictor:

Main high level API for person reidentification.

```python
from NewLUPersons import LUPersonPredictor

predictor = LUPersonPredictor(
    model_path="market.pth",
    device="cuda",
    batch_size=32,
    num_workers=4
)

#Extract features.
features = predictor.extract_features(images)

#Compute similarity.
similarities = predictor.compute_similarity(query_features, gallery_features)

#Rank by similarity.
rankings = predictor.rank_by_similarity(query_features, gallery_features, top_k=10)

#Retrieve similar persons.
results = predictor.retrieve_similar_persons(query_image, gallery_images, top_k=5)
```

### FeatureExtractor:

Lower level API for feature extraction.

```python
from NewLUPersons import FeatureExtractor

extractor = FeatureExtractor(model_path="market.pth")

#Extract from image paths.
features = extractor.extract_features(["image1.jpg", "image2.jpg"])

#Extract from directory.
features_dict = extractor.extract_features_from_directory("path/to/images")
```

## Attribution:

**Modified by:** Mridankan Mandal.

**Original Authors:** Dengpan Fu, Dongdong Chen, Jianmin Bao, Hao Yang, Lu Yuan, Lei Zhang, Houqiang Li, Dong Chen

**Original Paper:** [Unsupervised Pre-training for Person Reidentification](https://arxiv.org/abs/2104.14294) (CVPR 2021).

**Original Repository:** [LUPerson GitHub](https://github.com/DengpanFu/LUPerson).

## Examples:

See the `examples/` directory for complete examples:

- `load_and_extract_features.py`: Load model and extract features.
- `model_inference.py`: Perform inference and similarity search.

## Testing:

Run tests to verify the installation:

```bash
pytest tests/
```

## Compatibility

- **Operating System**: Windows 11, Linux, macOS
- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12
- **PyTorch**: 1.9+
- **CUDA**: 11.0+ (tested with 12.1)

## Citation:

If you use NewLUPersons in your research, please cite the original LUPerson paper:

```bibtex
@inproceedings{fu2021unsupervised,
  title={Unsupervised Pre-training for Person Reidentification},
  author={Fu, Dengpan and Chen, Dongdong and Bao, Jianmin and Yang, Hao and Yuan, Lu and Zhang, Lei and Li, Houqiang and Chen, Dong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Acknowledgements:

This library is built on top of:

- **[LUPerson](https://github.com/DengpanFu/LUPerson)**: Unsupervised pre-training for person reidentification (CVPR 2021).
- **[newFastReID](https://github.com/WhiteMetagross/newFastReID)**: Modernized FastReID library.
- **[FastReID](https://github.com/JDAI-CV/fast-reid)**: Original FastReID library for person reidentification.

## References

- **LUPerson Paper**: [Unsupervised Pre-training for Person Reidentification](https://arxiv.org/abs/2104.14294).
- **FastReID Paper**: [FastReID: A Pytorch Toolbox for General Instance Reidentification](https://arxiv.org/abs/2006.02631).
- **Original LUPerson Repository**: [https://github.com/DengpanFu/LUPerson](https://github.com/DengpanFu/LUPerson).


## Support

For issues, questions, or suggestions:

1. Check the [documentation](./InstallationAndSetup.md) and [examples](./examples/).
2. Review the [CodeBaseIndex](./CodeBaseIndex.md) for detailed API information.
3. Visit the [original LUPerson repository](https://github.com/DengpanFu/LUPerson) for model-specific questions.


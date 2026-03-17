# pose-estimation-recognition-utils-rtmlib

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**pose-estimation-recognition-utils-rtmlib** is the [rtmlib](https://github.com/Tau-J/rtmlib)-based variant of the [pose-estimation-recognition-utils](https://github.com/cobtras/pose-estimation-recognition-utils) package. It builds heavily on the core data structures and utilities of its parent package, providing a specialized implementation for high-performance 2D and 3D pose estimation using RTM models.

## Features

- **2D Pose Estimation**: Easy-to-use wrapper for RTM-based 2D pose detection in images and videos.
- **3D Pose Lifting**: Intelligent 2D-to-3D pose lifting using AI (from Hugging Face) or geometric heuristics.
- **Stereo 3D Estimation**: Extract 3D coordinates from synchronized left/right camera frames.
- **Model Management**: Automatic downloading and caching of lifting models from the Hugging Face Hub. Supports pre-configured models for 17 and 133 keypoints, as well as custom models via repo-ID.
- **Data Conversion**: Utility functions to convert RTM results to common skeleton data formats.
- **Visualization**: Built-in filtering and skeleton drawing utilities.

## Installation

```bash
pip install pose-estimation-recognition-utils-rtmlib
```

> [!NOTE] 
> For hardware acceleration during 3D lifting, please install the appropriate `onnxruntime` package for your system before installing `pose-estimation-recognition-utils-rtmlib`:
> - For **NVIDIA GPUs (CUDA/TensorRT)**: `pip install onnxruntime-gpu`
> - For **AMD GPUs (ROCm)**: `pip install onnxruntime-rocm`
> - If you only want to use the **CPU**, the standard `onnxruntime` is installed automatically.

### Requirements

- Python 3.10+
- `rtmlib`
- `pose-estimation-recognition-utils`
- `torch`
- `huggingface-hub`
- `opencv-python`
- `numpy`
- `tqdm`

## Quick Start

### 2D Pose Estimation

```python
import cv2
from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator2D

# Initialize estimator
estimator = RTMPoseEstimator2D(mode='performance', device='cpu')

# Process an image
image = cv2.imread('sample.jpg')
result = estimator.process_image(image)

print(f"Detected {result.num_persons} persons.")
```

### 3D Pose Lifting

```python
from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator3D

# Initialize 3D estimator (includes 2D detection and 3D lifting)
estimator_3d = RTMPoseEstimator3D(mode='performance', lifting_mode='ai')

# Process image to get 3D coordinates
result_3d = estimator_3d.process_image(image)
print(f"3D Keypoints shape: {result_3d.keypoints_3d.shape}")
```

## Running Tests

We use `pytest` for unit testing. To run the tests locally:

```bash
pip install pytest
python -m pytest
```

## Model Management

The library includes a `ModelLoader` class that handles the downloading and caching of pose lifting models. 

### Pre-configured Models
By default, the following models are available and pre-configured for use with `RTMLifting`:
- **17 Keypoints (COCO)**: `fhswf/rtm17lifting`
- **133 Keypoints (Whole Body)**: `fhswf/rtm133lifting`

### Custom Models
You can easily use your own models by providing a `special_model` path in the format `repo-id/filename`:

```python
from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator3D

# Use a custom lifting model from Hugging Face
estimator = RTMPoseEstimator3D(
    num_keypoints=133,
    lifting_mode='ai',
    special_model='your-username/your-model-repo/model.pth'
)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

Jonas David Stephan, Nathalie Dollmann, Sabine Dawletow, Benjamin Otto Ernst Bruch.

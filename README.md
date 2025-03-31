# YOLOv8 on AMD ROCm

This repository contains code for training and deploying a YOLOv8 object detection model using an AMD RX 7900 XT GPU with ROCm 6.3.3. The project includes scripts for training the model as well as for running inference on videos, including options to display class-specific colored bounding boxes.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Required Libraries and Versions](#required-libraries-and-versions)
- [Installation](#installation)
  - [1. ROCm 6.3.3 Installation](#1-rocm-633-installation)
  - [2. Creating and Activating a Virtual Environment](#2-creating-and-activating-a-virtual-environment)
  - [3. Installing Python Dependencies](#3-installing-python-dependencies)
  - [4. Installing ROCm-Enabled PyTorch](#4-installing-rocm-enabled-pytorch)
- [Running the Code](#running-the-code)
  - [Training](#training)
  - [Inference on Video](#inference-on-video)
    - [Standard Video Inference](#standard-video-inference)
    - [Colored Bounding Boxes by Class](#colored-bounding-boxes-by-class)
- [Challenges](#challenges)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project demonstrates how to train and deploy a YOLOv8 object detection model using an AMD RX 7900 XT GPU under ROCm on Linux. Due to limitations with ROCm support on Windows, the project was eventually run natively on Linux (dual-boot) to fully leverage GPU acceleration.

## Prerequisites

- **Operating System:** Linux (Dual-boot recommended for native GPU support)
- **GPU:** AMD RX 7900 XT
- **ROCm:** Version 6.3.3 (or compatible; see [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))
- **Python:** Version 3.10 (recommended for compatibility with ROCm-enabled builds)
- **Virtual Environment:** Either `conda` or Python `venv`

## Required Libraries and Versions

- **PyTorch:** ROCm-enabled build (e.g., version 2.6.0+rocm6.3 or similar)
- **torchvision:** ROCm-enabled version corresponding to your PyTorch build
- **torchaudio:** ROCm-enabled version
- **Ultralytics YOLO:** Version 8.3.91 (installed via pip)
- **OpenCV:** `opencv-python` (version 4.x)
- **numpy:** Version ≤ 2.1.1 (to satisfy Ultralytics YOLO dependency)
- Additional dependencies (installed via pip): filelock, fsspec, jinja2, networkx, etc.

## Installation

### 1. ROCm 6.3.3 Installation

Ensure that ROCm is properly installed on your system. Verify that your GPU is recognized by running:

/opt/rocm/bin/rocminfo | grep 'Name'


## 1. Create and Activate a Virtual Environment

You can use either **venv** or **conda**.

### Using `venv`:
```bash
python3 -m venv pytorch-rocm-env  
source pytorch-rocm-env/bin/activate
```

### Using `conda`:
```bash
conda create -n pytorch-rocm-env python=3.10  
conda activate pytorch-rocm-env
```

> After activation, your shell should show:  
> `(pytorch-rocm-env)` at the beginning of the prompt.

---

## 2. Install Python Dependencies

Upgrade pip and install required packages:
```bash
pip install --upgrade pip  
pip install ultralytics opencv-python numpy==2.1.1
```

---

## 3. Install ROCm-Enabled PyTorch

First, remove any existing PyTorch packages:
```bash
pip uninstall torch torchvision torchaudio
```

Then install the ROCm version:
```bash
pip install --pre --force-reinstall --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.3
```

### Verify the installation:
```bash
python -c "import torch; print('Version:', torch.__version__); print('torch.cuda.is_available():', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

You should see a version like `+rocm6.3` and a message showing that a GPU device is available.

---

## 4. Training the Model

### Prepare your dataset

Update the `data.yaml` file with paths to your training, validation, and test images.

### Start training

Run the training script:
```bash
python train_yolo.py
```

Model weights will be saved in:
runs/detect/trainX/weights/best.pt


---

## 5. Inference on Video

There are two scripts available for running inference on videos.

### Standard Inference

To annotate a video using your trained model:
# ```bash
python apply_model_to_video.py input_video.mp4 output_video.mp4 [conf_thresh]
# ```

- Replace `input_video.mp4` with your input file  
- Replace `output_video.mp4` with the output filename  
- `[conf_thresh]` is optional (default = `0.25`)

### Colored Bounding Boxes by Class

To draw boxes in specific colors for each class:
# ```bash
python video_detect_colored.py input_video.mp4 output_video.mp4 [conf_thresh]
# ```

- Blue boxes for **blue player**  
- Red boxes for **red player**  
- White boxes for **referee**  
- Green (default) for any other class

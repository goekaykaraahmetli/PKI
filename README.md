
# YOLOv8 on AMD ROCm

This repository contains code for training and deploying a YOLOv8 object detection model using an AMD RX 7900 XT GPU with ROCm 6.3.3. The project includes scripts for training the model as well as for running inference on videos, including options to display class-specific colored bounding boxes.

## Prerequisites

- **Operating System:** Linux (Dual-boot recommended for native GPU support)
- **GPU:** AMD GPU with ROCm Support (or NVIDIA GPU should also work)
- **ROCm:** Version 6.3.3 (or compatible; see [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))
- **Python:** Version 3.10 (recommended for compatibility with ROCm-enabled builds)
- **Virtual Environment:** Either `conda` or Python `venv`

## Required Libraries

- Install dependencies based on your hardware:
  - For **AMD GPUs (ROCm)**: use `requirementsAMD.txt`
  - For **NVIDIA GPUs**: use `requirementsNVIDIA.txt`

## Installation

### 1. ROCm 6.3.3 Installation (FOR AMD)

Ensure that ROCm is properly installed on your system. Verify that your GPU is recognized by running:

```bash
/opt/rocm/bin/rocminfo | grep 'Name'
```
## 1. Create and Activate a Virtual Environment

You can use either **venv** or **conda**. Make sure to use Python 3.10

```bash
sudo apt install python3.10 python3.10-venv -y
```

### Using `venv`:
```bash
python3.10 -m venv pytorch-rocm-env
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

Upgrade pip and install the required packages for your GPU.

### For AMD (ROCm):
```bash
pip install --upgrade pip
pip install -r requirementsAMD.txt
```
### For NVIDIA:
```bash
pip install --upgrade pip
pip install -r requirementsNVIDIA.txt
```
---

### Verify the installation:
```bash
cd DatasetPreperation
python testCuda.py
```
### For AMD:

You should see a version like `+rocm6.3` and a message showing that a GPU device is available.

---

## 4.1 Training the Model (YOLO) 

### (You can Skip Step 4.1 and 4.2 and go straight to Point 5 if you don't want to train from scratch, as weights are already present in this repository)

### Start training

Run the training script in /pipeline:
```bash
python train.py
```

Model weights will be saved in:
runs/detect/trainX/weights/best.pt

X in trainX represents the most recent run (so train5 for instance)


---

## Inference on Video

There are two scripts available for running inference on videos.

### Standard Inference

To annotate a video using your trained model:
```bash
python video_detect.py input_video.mp4 output_video.mp4
```

- Replace `input_video.mp4` with your input file  
- Replace `output_video.mp4` with the output filename  

### Colored Bounding Boxes by Class

To draw boxes in specific colors for each class:
```bash
python video_detect_colored.py input_video.mp4 output_video.mp4
```

- Blue boxes for **blue player**  
- Red boxes for **red player**  
- White boxes for **referee**  


## 4.2 Training the Model (CNN)

This section explains how to train a CNN classifier to work alongside YOLO for dual fighter classification.

### Steps to Prepare and Train the Model

1. **Download the Dataset**  
   - Get the Olympic Boxing dataset and extract it into the folder /DatasetPreperation:
   - Inside of /DatasetPreperation, execute this command: 
     ```bash
     gdown 
     ```

2. **Run Preprocessing Scripts**  
   These scripts will extract frames, detect fighters, and prepare the dataset (This step will take a while):
   - Note: If you want to use the model that was trained in 4.1, change the MODEL_PATH in extract_and_detect.py to the newest best.pt in Line 8. Changing the number of trainX should suffice (you can skip this step though)
   ```bash
   python extract_and_detect.py  
   python create_dataset.py  
   python split_dataset_TrainTestVal.py
   ```

4. **Train the CNN Classifier**  
   Train the model using:
   ```bash
   python train_dual_fighters.py
   ```

   After training, the best model weights will be saved as:
   best_dual_fighter.pth
   
## 5. Run YOLO + CNN Inference (This can be done without training first as repository already has pretrained .pth files)
To run both YOLO detection and CNN classification on a video, use:
```bash
python final_predict.py input_video.mp4 output_video.mp4
```

- Replace `input_video.mp4` with your input video  
- Replace `output_video.mp4` with your desired output filename  

> Note: This script is located in the `/pipeline` directory and combines YOLO detection with CNN-based class predictions for each detected fighter.

---

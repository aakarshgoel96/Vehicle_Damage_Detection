# Vehicle Damage Detection using YOLOv8

This project implements a YOLOv8-based model for detecting and classifying vehicle damages. It uses a Dockerized environment for easy setup and reproducibility.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Data Structure](#data-structure)
4. [Class Information](#class-information)
5. [Setup](#setup)
6. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
7. [Code Structure](#code-structure)
8. [Customization](#customization)


## Project Overview

This project uses YOLOv8, a state-of-the-art object detection model, to identify and classify various types of vehicle damage. The implementation is designed to work with the dataset, which includes images of damaged vehicles and their corresponding annotations.

## Prerequisites

- Docker
- Git (for cloning the repository)
- A CUDA-capable GPU (recommended for faster training)

## Data Structure

The expected data structure is as follows for yolov8 format
Place and run cocotoyolov8.py in data_preprocessing directory to your dataset folder to convert from coco to YOLO if its not already. It will generate following directory strucutre.

```
/path/to/your/data/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The `data.yaml` file should initially contain paths to the image directories. The script will automatically update it to include the annotation paths and the correct class information.

## Class Information

The model is trained to detect the following types of vehicle damage:

1. minor-dent
2. minor-scratch
3. moderate-broken
4. moderate-dent
5. moderate-scratch
6. severe-broken
7. severe-dent
8. severe-scratch

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/aakarshgoel96/Vehicle_Damage_Detection.git
   cd vehicle-damage-detection
   ```

2. Build the Docker image:
   ```
   docker build -t vehicle-damage-detection .
   ```

## Usage

### Training

To train the model :

```
docker run --rm -v /path/to/your/data:/app/data -v /path/to/save/model:/app/runs vehicle-damage-detection python train.py --epochs 100 --batch-size 16 --img-size 640
```

Adjust the following parameters as needed:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--img-size`: Input image size for the model

### Evaluation

To evaluate the trained model:

```
docker run --rm -v C:/Users/Goel/Desktop/Lensor/vehicle_damage_detection_dataset:/app/data -v C:/Users/Goel/Desktop/Lensor/models:/app/runs vehicle-damage-detection python eval.py  --model /path/to/trained_model/best.pt
```

Use following configuration to run above commands on gpus
```
docker run --rm --gpus '"device=0"' --shm-size=20g
```

## Code Structure

- `Dockerfile`: Defines the Docker environment
- `requirements.txt`: Lists Python dependencies
- `train.py`: Script for training the YOLOv8 model
- `evaluate.py`: Script for evaluating the trained model
- `utils.py`: Contains utility functions for data preparation and result visualization

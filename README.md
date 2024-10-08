# YOLO Object Detection with MNIST Dataset

This project implements a simplified YOLO (You Only Look Once) object detection model for detecting digits in the MNIST dataset using PyTorch. 

The model is trained using customized COCO-format annotations and can detect objects in images by predicting bounding boxes and class probabilities.

**Note:** The annotations used in this repository follow the format `[y, x, h, w]` (where `y` is the top coordinate, `x` is the left coordinate, and `h` and `w` are the height and width, respectively). The bounding boxes in YOLO are also based on this order.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project utilizes a custom implementation of the YOLO architecture to perform object detection on an MNIST object detection dataset. The model detects objects (digits) by outputting bounding boxes and class predictions in a grid-based structure. It is designed to handle COCO-style annotations and can be adapted to other datasets as needed.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/mnist-yolo-object-detection.git
    cd mnist-yolo-object-detection
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should include dependencies such as `torch`, `torchvision`, and any other libraries you are using.

## Usage

### Training

To train the model, run the following command:

```bash
python train.py
```

You can configure various settings like dataset paths, hyperparameters, and the number of epochs in the script.

Inference
Once the model is trained, you can use it for inference on new images:

```python
from model import YOLO
import torch
from utils import convert_yolo_to_coco_target

# Load model
model = YOLO(grid_size=5, num_bboxes=1, num_classes=10)
model.load_state_dict(torch.load('path_to_checkpoint.pth')['model_state_dict'])
model.eval()

# Perform inference
image = ...  # Load image here
output = model(image)

# Convert YOLO output to COCO format
coco_annotations = convert_yolo_to_coco_target(output, image_width=256, image_height=256)
```
Dataset Preparation
This project requires COCO-style annotations. You can use your custom dataset or the MNIST dataset (in a COCO-style annotation format).

Ensure you have the following structure for the dataset:
```
mnist_object_detection_dataset/
├── training/
│   ├── images/
│   │   ├── image_1.png
│   │   └── ...
│   └── annotations/
│       └── coco_annotations_with_labels.json
```
The images should be stored under the images/ directory and the COCO annotations JSON file should be placed in the annotations/ directory.

## Model Architecture
The YOLO model is composed of several key components:

- Convolutional Layers: The network starts with convolutional layers to extract spatial features from the input images.
- Residual Blocks: These are used to improve gradient flow and model accuracy.
- Adaptive Max Pooling: This layer adjusts the output to match the predefined grid size (e.g., 7x7).
- Output Layer: The final convolutional layer outputs bounding box coordinates, confidence scores, and class probabilities.
For detailed information about the architecture, refer to model.py.

## Training
The training script loads the dataset, performs preprocessing, and trains the model using Adam optimizer and a custom YOLO loss function. The key training steps are:

- Forward pass: The images are passed through the YOLO model to predict bounding boxes and class probabilities.
- Loss Calculation: The custom YoloLoss function computes the loss based on localization (bounding box coordinates), confidence, and classification loss.
- Backpropagation: The optimizer updates the model weights based on the computed loss.
- Model checkpoints are saved every 100 epochs in the model_checkpoints/ directory.

## Evaluation
To evaluate the model on new images or a test dataset, load the trained model and perform inference using the test images. The evaluation script will generate predictions and compare them against ground truth annotations.

## Contributing
If you'd like to contribute to this project, feel free to create a pull request or open an issue. Make sure to follow the coding guidelines and write clear commit messages.

## License
This project is licensed under the MIT License. See the LICENSE file for details.



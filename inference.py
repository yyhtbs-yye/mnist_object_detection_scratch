import torch
from model import YOLO  # Replace with your actual model class

import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from utils import convert_yolo_to_coco_target 
import numpy as np
from visualization import visualize_annotations

# Define the model parameters
S = 7  # Grid size
B = 2  # Number of bounding boxes per cell
C = 10  # Number of classes (in your case)

# Initialize your model
model = YOLO(grid_size=S, num_bboxes=B, num_classes=C)  # Make sure this matches your model class

# Load model weights
checkpoint_path = r'model_checkpoints\model_epoch_100.pth'  # Path to your model checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the desired device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Define image transformations (resize and normalize, similar to your model's training)
transform = transforms.Compose([
    transforms.ToTensor()  # Convert to PyTorch Tensor
])

root_dir = 'mnist_object_detection_dataset/test/images'  # Path to the folder containing images
ann_file = 'mnist_object_detection_dataset/test/annotations/coco_annotations_with_labels.json'  # Path to COCO annotations JSON

coco_dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=transform)

# Move the image to the device (CPU/GPU)
image = coco_dataset[560][0].to(device)
gt_annotations = coco_dataset[560][1]

# Forward pass through the model
with torch.no_grad():
    output = model(image.unsqueeze(0))  # Add a batch dimension to the image

# Assuming the output is the YOLO-style target (you might need to post-process it depending on your model)
output = output.squeeze(0)  # Remove batch dimension

# Convert the model output (YOLO format) to COCO format annotations
annotations = convert_yolo_to_coco_target(output, image_width=256, image_height=256, S=S, B=B, C=C)

# Print the annotations (optional)
print(annotations)

from PIL import Image

# Convert the image tensor back to a PIL image (unnormalized for visualization)
image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
image_np = (image_np * 255).astype(np.uint8)  # Denormalize
image_pil = Image.fromarray(image_np)

# Visualize the annotations on the image
class_labels = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9"]  # Replace with actual class labels if needed
visualize_annotations(image_pil, annotations, class_labels, save_path="inference.png")

a = 1
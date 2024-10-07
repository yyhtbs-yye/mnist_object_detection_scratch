
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import os

from model import YOLO
from loss import YoloLoss
from utils import convert_coco_to_yolo_target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

if __name__ == '__main__':
    # Define paths
    root_dir = 'mnist_object_detection_dataset/training/images'  # Path to the folder containing images
    ann_file = 'mnist_object_detection_dataset/training/annotations/coco_annotations_with_labels.json'  # Path to COCO annotations JSON

    # Define image transformations (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Create the COCO detection dataset
    coco_dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=transform)

    # Create a DataLoader to load the dataset in batches
    dataloader = DataLoader(coco_dataset, batch_size=8, shuffle=True,
                            num_workers=1, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    S, B, C = 5, 1, 10
    H, W = 256, 256

    # Set up model, loss function, and optimizer
    model = YOLO(grid_size=S, num_bboxes=B, num_classes=C)
    model = model.to(device)

    criterion = YoloLoss(S=S, B=B, C=C)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    save_dir = './model_checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training loop
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        total_loss = 0  

        for images, targets in dataloader:
            # Move data to device
            images = images.to(device)
            targets = torch.stack([convert_coco_to_yolo_target(target, H, W, S=S, B=B, C=C) for target in targets], axis=0).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # Save the model every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")

        avg_loss = total_loss / len(dataloader)
        print(f"EPOCH@{epoch + 1}, avg_loss: {avg_loss}")

import torch
import numpy as np

def convert_coco_to_yolo_target(annotations, image_width, image_height, S=7, B=2, C=10):
    """
    Convert COCO-style annotations to YOLO-style targets.
    
    Args:
    - annotations: List of annotations (each with a 'bbox' and 'category_id').
    - image_width: Width of the image.
    - image_height: Height of the image.
    - S: Grid size (default 7x7 for YOLO).
    - B: Number of bounding boxes per grid cell (default 2).
    - C: Number of classes.
    
    Returns:
    - A torch tensor of shape (S, S, B * (5 + C)) representing the YOLO target.
    """
    # Initialize target tensor with zeros. Shape is (S, S, B * (5 + C)):
    # For each grid cell, we have B bounding boxes, each with 5 values (x, y, w, h, confidence)
    # and C class probabilities.
    target = np.zeros((S, S, B * (5 + C)))

    # Normalize bounding box dimensions (x_min, y_min, width, height) and assign to grid cells
    for obj in annotations:
        bbox = obj['bbox']  # In COCO, bbox format is [x_min, y_min, width, height]
        class_id = obj['category_id']  # Class label (integer 0 to C-1)

        # Convert bbox from absolute pixel values to relative values (0-1)
        y_min, x_min, box_h, box_w = bbox
        y_min /= image_height
        x_min /= image_width
        box_h /= image_height
        box_w /= image_width

        # Find the center of the bounding box
        center_y = y_min + box_h / 2
        center_x = x_min + box_w / 2

        # Calculate which grid cell (i, j) the center of the object falls into
        cell_y = int(center_y * S)
        cell_x = int(center_x * S)

        # Ensure that the grid indices are within the bounds
        if cell_x >= S: cell_x = S - 1
        if cell_y >= S: cell_y = S - 1

        # Calculate the (x, y) offset within the grid cell
        cell_x_offset = center_x * S - cell_x
        cell_y_offset = center_y * S - cell_y

        # Try to assign the object to one of the available bounding box slots (B slots)
        assigned = False  # Variable to track if the object was assigned to a bounding box
        for b in range(B):
            # If no object has been assigned to this bounding box slot, assign it
            if target[cell_y, cell_x, b * (5 + C) + 4] == 0:  # Confidence score is 0 if not assigned
                # Bounding box (x_offset, y_offset, w, h, confidence)
                target[cell_y, cell_x, b * (5 + C):b * (5 + C) + 5] = np.array([cell_y_offset, cell_x_offset, box_h, box_w, 1.0])

                # Class probabilities (one-hot encoded)
                target[cell_y, cell_x, b * (5 + C) + 5 + class_id] = 1.0

                # Mark as assigned and break out of the loop
                assigned = True
                break
        
        # If no bounding box slot was available (i.e., B slots full), do nothing
        # This step ensures that if the grid cell is full, we ignore the new object

    return torch.tensor(target, dtype=torch.float32)

def convert_yolo_to_coco_target(yolo_target, image_width, image_height, confidence_threshold = 0.3, S=7, B=2, C=10):
    """
    Convert YOLO-style targets back to COCO-style annotations.
    
    Args:
    - yolo_target: Tensor of shape (S, S, B * (5 + C)) representing the YOLO target.
    - image_width: Width of the image.
    - image_height: Height of the image.
    - S: Grid size (e.g., 7x7 for YOLO).
    - B: Number of bounding boxes per grid cell.
    - C: Number of classes.
    
    Returns:
    - A list of COCO-style annotations, where each annotation is a dictionary with 'bbox' and 'category_id'.
    """
    annotations = []

    # Loop through each grid cell
    for i in range(S):
        for j in range(S):
            for b in range(B):              # Each grid cell can have up to B bounding boxes
                # Extract the bounding box coordinates and confidence score
                bbox_data = yolo_target[i, j, b * (5 + C): b * (5 + C) + 5].tolist()
                y_offset, x_offset, box_h, box_w, confidence = bbox_data

                # If confidence is less the threshold, skip this bounding box (no object detected)
                if confidence < confidence_threshold:
                    continue

                # Convert the (x_offset, y_offset) back to absolute coordinates
                center_y = (i + y_offset) / S  # center_y relative to the whole image
                center_x = (j + x_offset) / S  # center_x relative to the whole image

                # Convert width and height back to absolute dimensions
                box_h_abs = box_h * image_height
                box_w_abs = box_w * image_width

                # Convert center coordinates to top-left corner (COCO uses [x_min, y_min, width, height])
                y_min = (center_y - box_h / 2) * image_height
                x_min = (center_x - box_w / 2) * image_width

                # Extract the class probabilities
                class_probabilities = yolo_target[i, j, b * (5 + C) + 5: b * (5 + C) + 5 + C].tolist()
                class_id = np.argmax(class_probabilities)  # Class with the highest probability

                # Add the annotation in COCO format
                annotations.append({
                    'bbox': [y_min, x_min,  box_h_abs, box_w_abs],
                    'category_id': class_id
                })
    
    return annotations

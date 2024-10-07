import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_annotations(image, annotations, class_labels=None, save_path=None):
    """
    Visualize annotations on the image and optionally save the result.

    Args:
    - image (PIL.Image or np.ndarray): The image to visualize annotations on.
    - annotations (list): A list of dictionaries containing 'bbox' and 'category_id'.
    - class_labels (list): A list of class names. Defaults to None (just shows class IDs).
    - save_path (str): If provided, the image will be saved to this path.

    Returns:
    - Displays the image with annotations drawn on it, and optionally saves it to a file.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Create a drawing context on the image
    draw = ImageDraw.Draw(image)

    # Loop through annotations and draw them
    for annotation in annotations:
        bbox = annotation['bbox']
        class_id = annotation['category_id']

        # Convert bbox to [x_min, y_min, x_max, y_max]
        y_min, x_min, height, width = bbox
        x_max = x_min + width
        y_max = y_min + height

        # Draw rectangle (bounding box)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Draw class label (if provided)
        label = str(class_id) if class_labels is None else class_labels[class_id]
        draw.text((x_min, y_min), label, fill="red")

    # Optionally save the image
    if save_path:
        image.save(save_path)
        print(f"Image saved to {save_path}")

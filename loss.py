import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=10, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord  # Weight for coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object loss

    def forward(self, predictions, target):
        # predictions: [batch_size, S, S, B*(5 + C)]
        # target: [batch_size, S, S, B*(5 + C)]

        # Split the predictions into components
        pred_bboxes = predictions[..., :self.B * (5 + self.C)].view(-1, self.S, self.S, self.B, 5 + self.C)
        target_bboxes = target[..., :self.B * (5 + self.C)].view(-1, self.S, self.S, self.B, 5 + self.C)

        # Extract bounding box coordinates and class probabilities
        pred_box_coords = pred_bboxes[..., :5]  # [x, y, w, h, confidence] for each bbox
        pred_class_probs = pred_bboxes[..., 5:]  # Class probabilities for each bbox

        target_box_coords = target_bboxes[..., :5]  # Ground truth bounding boxes
        target_class_probs = target_bboxes[..., 5:]  # Ground truth class probabilities

        # Masks to identify where objects exist
        obj_mask = target_box_coords[..., 4] > 0  # Objectness mask [batch_size, S, S, B]
        noobj_mask = ~obj_mask  # Inverse of the object mask

        # 1. Localization Loss (x, y, w, h)
        # Only calculate for the cells where objects exist (obj_mask)
        coord_loss = self.lambda_coord * torch.sum(
            (target_box_coords[..., :4] - pred_box_coords[..., :4])**2 * obj_mask.unsqueeze(-1)
        )

        # 2. Confidence Loss
        # Object confidence loss (only where objects are present)
        obj_conf_loss = torch.sum(
            (target_box_coords[..., 4] - pred_box_coords[..., 4])**2 * obj_mask
        )
        # No-object confidence loss (where no objects exist)
        noobj_conf_loss = self.lambda_noobj * torch.sum(
            (target_box_coords[..., 4] - pred_box_coords[..., 4])**2 * noobj_mask
        )

        # 3. Classification Loss
        # Only for the bounding boxes where objects exist (obj_mask)
        class_loss = torch.sum(
            (target_class_probs - pred_class_probs)**2 * obj_mask.unsqueeze(-1)
        )

        # Total loss is the sum of the localization, confidence, and classification losses
        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss

        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_leaky_relu=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Activation function
        self.activation = nn.LeakyReLU(0.1) if use_leaky_relu else nn.ReLU()

    def forward(self, x):
        identity = x  # Save the input as the "skip connection"

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += identity
        out = self.activation(out)

        return out

class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=10):
        super(YOLO, self).__init__()
        self.grid_size = grid_size  # The size of the grid (e.g., 7x7)
        self.num_bboxes = num_bboxes  # Number of bounding boxes per grid cell
        self.num_classes = num_classes  # Number of object classes (0-9 for digits)

        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual blocks
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 64)
        self.res_block3 = ResidualBlock(64, 64)

        # More convolutional layers to deepen the network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling to reduce spatial dimensions

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling again

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Final downsampling
        )

        # Adaptive Max Pooling to match the grid size (SxS)
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d((grid_size, grid_size))

        # Final output convolution layer to predict (x, y, w, h, confidence) and classes
        self.output_layer = nn.Conv2d(512, num_bboxes * (5 + num_classes), kernel_size=1, stride=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional and batch normalization layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization for Conv layers (suitable for LeakyReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm layers: weight initialized to 1 and bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial Conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Additional convolutional layers to extract features
        x = self.conv_layers(x)

        # Adaptive Max Pooling to match grid size (e.g., 7x7)
        x = self.adaptive_maxpool(x)

        # Output layer to predict (x, y, w, h, confidence) and classes
        x = self.output_layer(x)

        # Reshape the output to (batch_size, grid_size, grid_size, B*(5 + C))
        x = x.permute(0, 2, 3, 1)  # Permute to [batch_size, grid_size, grid_size, num_bboxes*(5 + num_classes)]

        return x

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) implementation based on Mnih et al. 2013.
    Input: Stack of 4 grayscaled 84x84 frames -> Shape (batch_size, 4, 84, 84)
    Output: Expected Q-values for each action -> Shape (batch_size, num_actions)
    """
    def __init__(self, input_shape=(4, 84, 84), num_actions=3):
        super(DQN, self).__init__()
        
        in_channels = input_shape[0]
        
        # 1. Feature Extraction (Convolutional Layers)
        self.features = nn.Sequential(
            # First Layer: Scans with 32 filters, size 8x8, stride 4
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Second Layer: Scans with 64 filters, size 4x4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Third Layer: Scans with 64 filters, size 3x3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 2. Compute Flattened Size
        conv_out_size = self._get_conv_out(input_shape)
        
        # 3. Decision Making (Fully Connected Layers)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        """Passes a dummy tensor to automatically calculate flattened conv output size."""
        dummy = torch.zeros(1, *shape)
        dummy_out = self.features(dummy)
        return int(dummy_out.view(1, -1).size(1))

    def forward(self, x):
        """
        Runs the forward pass.
        Expects input shape: (batch_size, channels, height, width)
        Returns: expected Q-values of shape (batch_size, num_actions)
        """
        conv_out = self.features(x)
        flat_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(flat_out)

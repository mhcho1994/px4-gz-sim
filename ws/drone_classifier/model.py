import torch
import torch.nn as nn

class DroneTrajectoryCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten() # flatten to 1D
        
        # 3. Fully Connected Layer
        self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.relu3 = nn.ReLU()
        
        # Two final outputs (PX4 Possibility Score vs ArduPilot Possibility Score)
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        # shape of x: (batch_size, number of features, length of sequence)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        out = self.fc2(x) # final score (Logits)
        return out

import torch.nn as nn
import torch.nn.functional as F

class Classifer_mlp(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.05):
        super(Classifer_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Initialize the dropout layer

    def forward(self, x):
        # First layer with ReLU
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout(x1)  # Apply dropout after activation

        # Add a residual connection after the second layer
        x2 = F.relu(self.fc2(x1))
        x2 = x2 + x1  # Non-inplace operation for safe gradient computation
        x2 = self.dropout(x2)  # Apply dropout

        # Another layer with ReLU
        x3 = F.relu(self.fc3(x2))
        x3 = x3 + x2  # Non-inplace operation
        x3 = self.dropout(x3)  # Apply dropout

        # Output layer, without non-linearity if you're using CrossEntropyLoss later
        x4 = self.fc4(x3)
        return x4
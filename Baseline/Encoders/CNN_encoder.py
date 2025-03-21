
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNEncoder(nn.Module):
    def __init__(self, seq_len, latent_dim):
        '''
            [b,steps,x,y]->[b,latent_dims]
        '''
        super(CNNEncoder, self).__init__()
        self.steps = seq_len
        self.latent_dim = latent_dim
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(256 * self.steps, latent_dim)  # 多步骤信息整合

    def forward(self, x):
        # Reshape to treat each step as a separate batch sample
        batch_size = x.size(0)

        x = x.view(batch_size * self.steps, 1, x.size(2), x.size(3))  # Assuming grayscale input
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.global_pool(x)  # Spatial dimensions are now 1x1
        x = x.view(batch_size, self.steps * 256)  # Flatten and prepare for fully connected
        x = self.fc(x)
        return x

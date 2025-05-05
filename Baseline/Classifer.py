import torch.nn as nn
import torch.nn.functional as F
import torch
class Classifer_mlp(nn.Module):

    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        h = F.gelu(self.norm1(self.fc1(x)))
        h = self.dropout(h)

        h2 = F.gelu(self.norm2(self.fc2(h)))
        h = h + h2
        h = self.dropout(h)

        h3 = F.gelu(self.norm3(self.fc3(h)))
        h = h + h3
        h = self.dropout(h)

        return self.fc_out(h)


class Classifier_LSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=4, num_layers=2, dropout_rate=0.01):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # MLP with residuals
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.out = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        # shortcut projection if needed
        self.shortcut = nn.Linear(hidden_dim, 512) if hidden_dim != 512 else nn.Identity()

    def forward(self, x):  # x: [B, T, input_dim]
        _, (h_n, _) = self.lstm(x)       # h_n: [num_layers, B, hidden_dim]
        agg = h_n[-1]                    # [B, hidden_dim]

        # First projection (with residual shortcut if needed)
        x = F.relu(self.fc1(agg))
        x = self.dropout(x)
        shortcut = self.shortcut(agg)

        # Residual block 1
        x1 = F.relu(self.fc2(x) + x)
        x1 = self.dropout(x1)

        # Residual block 2
        x2 = F.relu(self.fc3(x1) + x1)
        x2 = self.dropout(x2)

        # Output layer with optional skip
        logits = self.out(x2 + shortcut)  # optional residual to input
        return logits

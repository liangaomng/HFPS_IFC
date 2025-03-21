from neuralop.models import FNO

import torch.nn as nn
import torch.nn.functional as F
class OperatorEncoder(nn.Module):
    def __init__(self, seq_len, latent_dim,n_modes=(128,128),hidden_channels=32,x=128,y=512):
        '''
            [b,steps,x,y]->[b,latent_dims],n_modes=(128,128)
        '''
        super(OperatorEncoder, self).__init__()
        self.steps = seq_len
        self.latent_dim = latent_dim
        
        # operators layers
        self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                              in_channels= seq_len, out_channels=1)

        # Fully connected layer
        self.fc = nn.Linear(1 *x* y , latent_dim)  # 多步骤信息整合

    def forward(self, x):
       
        # Reshape to treat each step as a separate batch sample
        x = self.operator(x)
        b,t,sp_x,sp_y =  x.shape
        x = x.reshape(-1,t*sp_x*sp_y)
        x = F.elu(self.fc(x))
        return x

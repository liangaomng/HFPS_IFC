from neuralop.models import FNO
import torch
import torch.nn as nn
import torch.nn.functional as F
class OperatorEncoder(nn.Module):
    def __init__(self, seq_len, latent_dim,n_modes=(32,32),hidden_channels=1,x=128,y=512):
        '''
            [b,steps,x,y]->[b,latent_dims],n_modes=(128,128) 
        '''
        super(OperatorEncoder, self).__init__()
        self.steps = seq_len
        self.latent_dim = latent_dim
        
        # operators layers
        self.operator = FNO(n_modes=n_modes, hidden_channels=hidden_channels,
                              in_channels= seq_len, out_channels=1)
        self.norm = nn.LayerNorm(1 *x* y)
        # Fully connected layer
        self.fc = nn.Linear(1 *x* y , latent_dim)  # 多步骤信息整合

    def forward(self, x):
       
        # Reshape to treat each step as a separate batch sample
        x = self.operator(x)
        b,t,sp_x,sp_y =  x.shape
        x = x.reshape(-1,t*sp_x*sp_y)
        x = self.norm(x)         # LayerNorm
        x = F.elu(self.fc(x))
        return x

class Gurvan_OperatorEncoderLSTM(nn.Module):
    def __init__(self, seq_len, latent_dim, n_modes=(64, 64), hidden_channels=1, x=128, y=512):
        '''
            [b,steps,x,y]->output_seq.shape == [B, T, hidden_size]
'''
        super().__init__()
        self.operator = FNO(n_modes=n_modes,
                            hidden_channels=hidden_channels,
                            in_channels=1, out_channels=1)
        self.x, self.y = x, y
        self.latent_dim = latent_dim

        self.fc_frame = nn.Linear(x * y, latent_dim)

        # LSTM instead of GRU
        self.rnn = nn.LSTM(input_size=latent_dim,
                           hidden_size=latent_dim,
                           batch_first=True)

class Gurvan_OperatorEncoder(nn.Module):
    def __init__(self, seq_len, latent_dim, n_modes=(64, 64), hidden_channels=12, x=128, y=512):
        '''
        For each step, the operator is applied independently.
        Input: [b, t, x, y]
        Output: [b, t, latent_dim]
        '''
        super(Gurvan_OperatorEncoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.x, self.y = x, y

        # FNO applied to each time step independently
        self.operator = FNO(n_modes=n_modes,
                            hidden_channels=hidden_channels,
                            in_channels=1,  # Only 1 step at a time
                            out_channels=1)

        # Fully connected layer to project [b, x*y] -> [b, latent_dim]
        self.fc = nn.Linear(x * y, latent_dim)

    def forward(self, x):
        # x: [b, t, x, y]
        B, T, X, Y = x.shape
        latents = []

        for t in range(T):
            xt = x[:, t:t+1, :, :]  # [B,1,x,y]
            fno_out = self.operator(xt)  # [B,1,x,y]
            fno_out = fno_out + xt       # residual
            latent = self.fc(fno_out.view(B, -1))
            latents.append(latent)

        # stack over time -> [b, t, latent_dim]
        stack = torch.stack(latents, dim=1)
        return stack


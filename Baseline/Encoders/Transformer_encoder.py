

import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerEncoderModel(nn.Module):
    def __init__(self, seq_len, feature_size, latent_dim=1024,nheads=1,num_layers=1):
        super(TransformerEncoderModel, self).__init__()
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.latent_dim = latent_dim

        # 将输入展平
        self.flatten = nn.Flatten(start_dim=2)  # 将128 x 256展平

        # 定义 Transformer 的配置
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=nheads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)


        # 输出层
        self.fc = nn.Linear(seq_len * feature_size, latent_dim)

    def forward(self, x):
        # 调整输入尺寸 [batch, 10, 130, 258] -> [batch, 10, 33420]
        x = self.flatten(x)  # 假设每个时间步展平后的维度是33420

        # Transformer 编码
        x = x.permute(1, 0, 2)  # 调整为 [seq_len, batch, feature_size]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 调整回 [batch, seq_len, feature_size]

        # 输出层
        x = x.reshape(x.shape[0], -1)  # 展平所有特征
        x = self.fc(x)
        return x
class Classifier(nn.Module):
    '''
        return logits rather softmax
    '''
    def __init__(self, input_dim, num_classes=3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)  # 批归一化
        self.fc2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x= self.fc3(x)
        return x
class Transformer_FullModel(nn.Module):
    def __init__(self, input_dim, latent_dim,nums_classifers):
        super(Transformer_FullModel, self).__init__()
        # 假设 d_model 是输入和输出特征的维数，nhead 是多头注意力的头数
        seq_len= input_dim[0]
        feature_size = input_dim[1]*input_dim[2]
        self.attention =  TransformerEncoderModel(seq_len, feature_size, latent_dim)
        self.classifiers = nn.ModuleList([Classifier(latent_dim) for _ in range(nums_classifers)])

    def forward(self, x):
        latent_vector = self.attention(x)
        outputs = [classifier(latent_vector) for classifier in self.classifiers]
        outputs = torch.stack(outputs,dim=1)
        return outputs
     
     
if __name__ == "__main__":
   model = Transformer_FullModel(input_dim=[50,64,64], latent_dim=1024,nums_classifers=4)
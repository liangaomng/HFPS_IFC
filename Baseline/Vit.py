import torch.nn as nn
import torch.nn.functional as F
import torch
from vit_pytorch.vit_3d import ViT
class Vit_classifer(nn.Module):
    def __init__(self,num_classes= 3,image_size= 100,patch_size= 4, num_frames = 10):             # number of frames):
        super(Vit_classifer, self).__init__()
        self.vit = ViT(
                        image_size = image_size,
                        image_patch_size = patch_size,
                        frames = num_frames,
                        frame_patch_size= 1,
                        num_classes = num_classes,
                        dim = 128,
                        depth = 1,
                        heads = 2,
                        mlp_dim = 128,
                        dropout = 0.1,
                        channels =1,
                        emb_dropout = 0)#video = torch.randn(4, 1, 16, 128, 128) # (batch, channels, frames, height, width)
 
    def forward(self,x):
       
        x = torch.unsqueeze(x, 1)  # x[b, 1, t, x, y] channel
        out = self.vit(x) #(b,classes)
        return out
    
if __name__ == "__main__":
    v = ViT(
        image_size = 64,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)
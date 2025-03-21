
import torch.nn as nn
import torch.nn.functional as F
import torch
class IFC_model(nn.Module):
   
   def __init__(self,encoder:nn.Module,classifers:nn.ModuleList,ifc_type="Encoder_based"):
      super(IFC_model, self).__init__()
      self.encoder = encoder
      self.classifers = classifers #every classifier output logits not softmax layer [b,classes]
      self.type = ifc_type #or "Vit_based"
   
   def forward_encoder_classifer(self, x):
      '''
         vit did not need encoder
      '''
      self.latent = self.encoder(x)
      results = []
      for classifier in self.classifers:
            results.append(classifier(self.latent))
      # Stack the results to get shape [b, number of classifiers, classes]
      return torch.stack(results, dim=1)

   def forward_vit(self,x):
      results = []
      for classifier in self.classifers:
            results.append(classifier(x))
      # Stack the results to get shape [b, number of classifiers, classes]
      return torch.stack(results, dim=1)
   
   def forward(self,x):
      if self.type == "Encoder_based":
         return self.forward_encoder_classifer(x)
      elif self.type == "Vit_based":
         return self.forward_vit(x)

   
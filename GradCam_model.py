import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision as trv
from PIL import Image
import cv2


class GradResnet50(nn.Module):
    def __init__(self):
        super(GradResnet50, self).__init__()
        self.model = trv.models.resnet50(weights=trv.models.ResNet50_Weights.IMAGENET1K_V2)
        self.gradients = None
        self.transform = self.transform_obj()
        
    def transform_obj(self):
        transform = trv.transforms.Compose([
            trv.transforms.Resize(256),
            trv.transforms.CenterCrop(224),
            trv.transforms.ToTensor(),
            trv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        return transform
    
    def preprocess(self, img):
        img = self.transform(img)
        input_tensor = torch.unsqueeze(img, axis=0)
        return input_tensor
        
    def forward(self, x):
        x = self.preprocess(x)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        h = x.register_hook(self.activations_hook)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x 
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations(self, x):
        x = self.preprocess(x)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
    
    def Normalize_img(self, heatmap):
        heatmap = torch.mean(heatmap, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        return heatmap
    
    def postprocess(self, heatmap):
        heatmap = trv.transforms.Resize(224)(torch.unsqueeze(heatmap, axis=0))
        heatmap = torch.nn.functional.pad(heatmap, (16, 16, 16, 16, 0, 0), value=0.)
        heatmap = heatmap.squeeze()
        return heatmap
    
    def Gradheatmap(self, x):
        self.zero_grad()
        pred = self.__call__(x)
        _, index = torch.max(pred, axis=1)
        pred[:, index].backward()
        
        gradients = self.gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.get_activations(x).detach()
        
        batch_size = activations.shape[0]
        depth = activations.shape[1]
        
        heatmap = activations * torch.reshape(pooled_gradients, (batch_size, depth, 1, 1))
        # for i in range(2048):
        #     activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = self.Normalize_img(heatmap)
        
        heatmap = self.postprocess(heatmap)
        
        return heatmap
    

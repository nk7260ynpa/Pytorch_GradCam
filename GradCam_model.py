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
        self.img_width = 0
        self.img_height = 0
        self.eval()
    
    #Original Preprocess in ResNet50
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
        
        self.img_width, self.img_height = x.size
        
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
    
    #Resize to 224 and Pad to 256
    def postprocess(self, heatmap):
        heatmap = trv.transforms.Resize(224)(torch.unsqueeze(heatmap, axis=0))
        heatmap = torch.nn.functional.pad(heatmap, (16, 16, 16, 16, 0, 0), value=0.)
        heatmap = heatmap.squeeze()
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (self.img_width, self.img_height))
        return heatmap
    
    #Convert to RGB
    def Colorize_heatmat(self, heatmap):
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap
    
    #Gen heatmap
    def Gradheatmap(self, x):
        self.zero_grad()
        pred = self.__call__(x)
        _, index = torch.max(pred, axis=1)
        pred[:, index].backward()
        
        gradients = self.gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.get_activations(x).detach()
        
        heatmap = activations * torch.reshape(pooled_gradients, (*activations.shape[:2], 1, 1))
        heatmap = self.Normalize_img(heatmap)
        heatmap = self.postprocess(heatmap)
        heatmap = self.Colorize_heatmat(heatmap)
        
        return heatmap
    

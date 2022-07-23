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
        
    def forward(self, x):
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
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
    

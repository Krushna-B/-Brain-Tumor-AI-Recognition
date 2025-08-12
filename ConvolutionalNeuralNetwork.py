import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import sys
from dataset import brain_ImageData
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
      # Class constuctor
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6 ,kernel_size=5),       #First Convultional Network Layer and then Tanh function Layer and averagePool function Layer 
                nn.BatchNorm2d(6),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=5),

                nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),          #Second Convultional Network Layer and then Tanh function Layer and averagePool function Layer
                 nn.BatchNorm2d(16),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=5)
            )

        self.fc_model = nn.Sequential(
                nn.Linear(in_features=256, out_features=120),    #First Connection Network Layer and Tanh Function Layer
                 nn.Dropout(0.25),
                nn.Tanh(),

                nn.Linear(in_features=120,out_features=84),      #Second Connective Network Layer and Tanh Function Layer
                nn.Tanh(),

                nn.Linear(in_features=84,out_features=1)       #Last Connective Network Layer and 1 output neuron returing healthy or tumor
            ) 
        
    def forward(self,input):              # Input Data passes through convolutional layers and then connective layers and then through sigmoid function
                                                                            # Sigmoid Function returns a value between 0 and 1
            input = self.cnn_model(input)
            input = input.view(input.size(0), -1)
            input = self.fc_model(input) 
            input = F.sigmoid(input)
                
            return input






from copy import copy
from fileinput import filename
from pickle import TRUE
from xml.dom import HierarchyRequestErr
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import cv2
from PIL import Image
import argparse

#model
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1=nn.Conv2d(1, 32, 5, padding=1, stride=2)
        self.pool=nn.MaxPool2d(2,2)

        self.conv2=nn.Conv2d(32, 16, 5)
        self.fc1=nn.Linear(16*1*1, 128)
        self.fc2=nn.Linear(128, 64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        #print(x.shape)
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(x.shape[0], -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        #x=F.softmax(x, dim=1)
        return x

model=Network()

#cuda
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)

#loading model
state_dict=torch.load('checkpoint.pth')
model.load_state_dict(state_dict)
print(model.load_state_dict(state_dict))

def predict(pth):
    img=cv2.imread(pth)
    rgb_img=img.copy()
    rgb_img=cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gray_img=rgb_img.copy()
    gray_img=cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    retval ,threshold=cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    sorted_contours=sorted(contours, key=cv2.contourArea, reverse=True)
    clean_contours=sorted_contours[0:4]
    clean_contours = sorted(clean_contours, key=lambda x: cv2.boundingRect(x)[0])
    for c in clean_contours:
        x,y,w,h = cv2.boundingRect(c)
        rect = cv2.rectangle(rgb_img, (x,y), (x+w, y+h), (0,255,0), 2)
        
    
    plt.imshow(rect)
    plt.show()

predict('img/2022.jpg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size=(1, 32, 32)):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, (5, 5)) # padding = (F-1)/2 (to keep the size constant) 
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2) # 32
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        
        self.fc1 = nn.Linear(7744, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        fts = self.relu(self.conv1(x))
        fts = self.relu(self.conv2(fts))
        fts = self.pool2(fts)
        fts = self.relu(self.conv3(fts))
        
        flat_fts = fts.view(-1, 7744)
        flat_fts = self.relu(self.fc1(flat_fts))
        flat_fts = self.relu(self.fc2(flat_fts))
        out_fts = flat_fts
        out = self.fc3(flat_fts)
        return out_fts
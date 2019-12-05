import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size=(1, 32, 32)):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5)), # padding = (F-1)/2 (to keep the size constant) 
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
        )
        
        self.flat_fts = self.get_flat_fts(input_size, self.features)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
    
    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return filterlat_fts

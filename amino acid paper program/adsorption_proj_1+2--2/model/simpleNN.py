import torch
import torch.nn as nn

class FCNet(torch.nn.Module):
    def __init__(self, feature_dim1, feature_dim2):
        super(FCNet, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = nn.Linear(feature_dim1*feature_dim2, 16)
        self.linear2 = nn.Linear(16, 2) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
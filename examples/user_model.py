import torch
from torch import nn


class PrivateInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(3,2)
        self.output=nn.Linear(2,1)

    def forward(self,x):
        return self.output(torch.relu(self.input(x)))


def create_model():
    torch.manual_seed(123)
    return PrivateInferenceModel()

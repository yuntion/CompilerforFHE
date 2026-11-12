"""Small real models for baseline verification, with no implicit approximation."""

def make_demo(name):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(7)

    class Square(nn.Module):
        def forward(self,x): return x*x

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Conv2d(1,1,2)
            self.fc=nn.Linear(1,2)
        def forward(self,x):
            x=self.conv(x); x=x*x
            return self.fc(F.avg_pool2d(x,2).flatten(1))

    class ReluCore(nn.Module):
        def forward(self,x): return F.relu(1.5*x+0.125)

    class MaxPool(nn.Module):
        def forward(self,x): return F.max_pool2d(x,2)

    class PolynomialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("wq",torch.tensor([[0.5,0.125],[-0.25,0.5]]))
            self.register_buffer("wk",torch.tensor([[0.25,-0.5],[0.5,0.25]]))
            self.register_buffer("wv",torch.tensor([[0.5,0.5],[-0.125,0.25]]))
        def forward(self,x):
            q,k,v=x@self.wq,x@self.wk,x@self.wv
            scores=(q@k.transpose(-1,-2))*0.5
            weights=(scores*scores+1.0)*0.25
            return weights@v

    if name=="affine":
        model=nn.Linear(3,2); x=torch.tensor([[-0.5,0.25,0.75]])
    elif name=="square-mlp":
        model=nn.Sequential(nn.Linear(3,3),Square(),nn.Linear(3,2)); x=torch.tensor([[-0.5,0.25,0.75]])
    elif name=="tiny-cnn":
        model=TinyCNN(); x=torch.linspace(-0.5,0.5,16).reshape(1,1,4,4)
    elif name=="poly-attention":
        model=PolynomialAttention(); x=torch.tensor([[[-0.5,0.25],[0.75,-0.25]]])
    elif name=="relu":
        model=ReluCore(); x=torch.tensor([[-0.5,0.75]])
    elif name=="relu-mlp":
        model=nn.Sequential(nn.Linear(3,2),nn.ReLU(),nn.Linear(2,2)); x=torch.tensor([[-0.5,0.25,0.75]])
    elif name=="maxpool":
        model=MaxPool(); x=torch.tensor([[[[-0.75,0.5],[1.25,-0.25]]]])
    else: raise ValueError(f"unknown demo {name}")
    return model.eval(),(x,)


DEMOS=("affine","square-mlp","tiny-cnn","poly-attention","relu","relu-mlp","maxpool")

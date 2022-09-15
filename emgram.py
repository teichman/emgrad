import torch
import torch.nn as nn
import math
import argparse
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F


############################################################
# Black box functions to choose from
############################################################

def bbf1(x):
    z = 10 * x + 5
    z = z.repeat(1, 2)
    return z

def bbf2(x):
    z = 10 * x**2 + 5
    z = z.repeat(1, 2)
    return z

class StatefulBBF:
    def __init__(self):
        self.vals = torch.rand(64)
        
    def __call__(self, x):
        return bbf2(x) * self.vals

bbf3 = StatefulBBF()
    
def bbf2(x):
    lengths = (x * x).sum(axis=1).sqrt().unsqueeze(1)
    x = x / lengths
    x = x.repeat(1, 2)
    return x


############################################################
# Implementation of empirical gradient backward pass
############################################################

class EmpiricalGradientWrapper(torch.autograd.Function):
    @staticmethod
    def black_box_fn(x):
        raise NotImplementedError
    
    @staticmethod
    def forward(ctx, x):
        y = EmpiricalGradientWrapper.black_box_fn(x)
        # We don't want to save these when forwarding for empirical gradient estimation.
        if ctx:
            ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        batch_size = x.shape[0]
        input_size = x.shape[1]
        output_size = grad_output.shape[1]
        x = x.type(torch.float64)
        
        # grad_output is (batch_size, output_size)
        # We need to return gradient matrix of (batch_size, input_size)
        # grad_local[i, j, k] is the empirical estimate of the derivative of the jth output wrt the kth input in the ith batch.
        grad_local = torch.zeros((batch_size, output_size, input_size), dtype=torch.float64)
        for _ in range(args.num_reps):
            for idx in range(input_size):
                x[:, idx] += args.eps
                y2 = EmpiricalGradientWrapper.forward(None, x)
                x[:, idx] -= args.eps
                grad_local[:, :, idx] += (y2 - y) / args.eps
                
        grad_local /= args.num_reps  # Average across reps.

        # dout/din * dL/dout for all batch elements.
        grad_in = (grad_local.permute(0, 2, 1).type(torch.float32) @ grad_output.unsqueeze(2)).squeeze()
        return grad_in

# Glue
class BlackBoxLayer(nn.Module):
    def __init__(self, black_box_fn):
        super().__init__()
        self.egw = EmpiricalGradientWrapper()
        EmpiricalGradientWrapper.black_box_fn = black_box_fn  # This is really gross but whatever

    def forward(self, x):
        return self.egw.apply(x)

    def backward(self, ctx, grad_output):
        assert False


############################################################
# Model
############################################################
        
class ModelWithBlackBoxLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # Deep net implementation of NWP parameterizations
            nn.Linear(32, 64),  # Potentially high-d input, can be anything
            nn.ReLU6(),
            nn.Linear(64, 32),
            
            # NWP black box.
            # Put whatever gross nightmare of a function you want in here, without a defined backwards method.
            BlackBoxLayer(bbf1),
            
            # Weather super resolution & "style transfer"
            nn.Linear(64, 128),
            nn.ReLU6(),
            nn.Linear(128, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.ReLU6(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)

    
############################################################
# Training
############################################################
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--num-reps", type=int, default=1,
                        help="Number of reps of empirical gradient estimation per backprop step.")
    args = parser.parse_args()    

    # Super simple dataset: Just return a random tensor.
    # Deep net is trained to reproduce it as output.
    class RandomDataset(IterableDataset):
        def __init__(self):
            super().__init__()
        def __iter__(self):
            return self
        def __next__(self):
            return torch.round(torch.rand(32, dtype=torch.float32))
        
    ds = RandomDataset()
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    torch.autograd.set_detect_anomaly(True)
    
    model = ModelWithBlackBoxLayer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)

    model.train(True)
    for step, inputs in enumerate(dl):
        optimizer.zero_grad()
        probs = model(inputs)
        loss = ((probs - inputs)**2).mean()
        loss.backward()
        optimizer.step()

        num_errors = (probs.round() - inputs).abs().sum().item()
        acc = 1.0 - num_errors / inputs.numel()
        print(f"{step=} loss={loss.item():.4f} {acc=:.4f}")

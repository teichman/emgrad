import torch
import torch.nn as nn
import math
import argparse
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F

class BlackBoxLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn.apply

    def forward(self, x):
        return self.fn(x)

class BlackBoxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 10 * input + 5

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_local = torch.ones(input.shape, dtype=torch.float32) * 10
        return grad_output * grad_local


class RandomDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        return torch.round(torch.rand(8, dtype=torch.float32))

    
class ModelWithBlackBoxLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # Parameterizations
            nn.Linear(8, 16),
            nn.ReLU6(),
            nn.Linear(16, 32),
            nn.ReLU6(),
            nn.Linear(32, 64),  # 64 values put into NWP
            # NWP black box layer
            BlackBoxLayer(BlackBoxFunction()),
            # Weather super resolution & fine tuning layer
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Linear(32, 16),
            nn.ReLU6(),
            nn.Linear(16, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stack(x)
        return x

    
def main():
    ds = RandomDataset()
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    model = ModelWithBlackBoxLayer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train(True)
    for inputs in dl:
        optimizer.zero_grad()
        probs = model(inputs)
        loss = ((probs - inputs)**2).mean()
        loss.backward()
        optimizer.step()

        num_errors = (probs.round() - inputs).abs().sum().item()
        acc = 1.0 - num_errors / inputs.numel()
        print(f"{loss.item()=:.4f}  {acc=:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "--bs", type=int, default=1)
    parser.add_argument("-g", "--gpu", type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()    

    device = torch.device("cpu")
    if args.gpu is not None:
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using device {device}")

    main()

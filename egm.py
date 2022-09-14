import torch
import torch.nn as nn
import math
import argparse
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F

class BlackBoxFunction(torch.autograd.Function):
    @staticmethod
    def bbf(x):
        # Put whatever nightmare of code you want here.
        return 10 * x + 5
    
    @staticmethod
    def forward(ctx, x):
        y = BlackBoxFunction.bbf(x)
        if ctx:
            ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        batch_size = x.shape[0]
        input_size = x.shape[1]
        output_size = grad_output.shape[1]
        
        # grad_output is (batch_size, output_size)
        # We need to return gradient matrix of (batch_size, input_size)
        # grad_local is (output_size, input_size)
        # grad_local[i, j] is the empirical estimate of the derivative of the jth output wrt ith input.
        grad_local = torch.zeros((output_size, input_size), dtype=torch.float32)
        for _ in range(args.num_reps):
            for idx in range(input_size):
                x[:, idx] += args.eps
                y2 = BlackBoxFunction.forward(None, x)
                x[:, idx] -= args.eps
                grad_local[idx, :] += (y2 - y).sum(axis=0) / batch_size  # Average gradient across batch
                
        grad_local /= args.num_reps
        grad_local /= args.eps

        return grad_output @ grad_local
    
class PlaceholderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 10 * x + 5

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_local = torch.ones(x.shape, dtype=torch.float32) * 10  # dout/din
        # grad_output is dL/dout
        # We need to return dL/din = dL/dout * dout/din.
        # It'll have one value per input element per batch.
        # Shape of return tensor should be (batch_size, input_size)
        result = grad_output * grad_local
        assert result.shape == x.shape
        return result

class BlackBoxLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn.apply

    def forward(self, x):
        return self.fn(x)
    
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
            #BlackBoxLayer(BlackBoxFunction()),
            BlackBoxLayer(PlaceholderFunction()),
            # Weather super resolution & fine tuning layer
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Linear(32, 16),
            nn.ReLU6(),
            nn.Linear(16, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--num-reps", type=int, default=5,
                        help="Number of reps of empirical gradient estimation per backprop step.")
    args = parser.parse_args()    

    # Super simple dataset: Just return us some random 0s and 1s.
    class RandomDataset(IterableDataset):
        def __init__(self):
            super().__init__()
        def __iter__(self):
            return self
        def __next__(self):
            return torch.round(torch.rand(8, dtype=torch.float32))
        
    ds = RandomDataset()
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    model = ModelWithBlackBoxLayer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

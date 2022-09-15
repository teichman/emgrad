import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, IterableDataset


############################################################
# Implementation of empirical gradient backward pass
############################################################

class EmgradWrapper(torch.autograd.Function):
    @staticmethod
    def black_box_fn(x):
        raise NotImplementedError
    
    @staticmethod
    def forward(ctx, x):
        y = EmgradWrapper.black_box_fn(x)
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
                y2 = EmgradWrapper.forward(None, x)
                x[:, idx] -= args.eps
                grad_local[:, :, idx] += (y2 - y) / args.eps
                
        grad_local /= args.num_reps  # Average across reps.

        # Compute analytic gradient for comparison.
        if torch.rand(1) < args.analytic_comparison_frac:
            gla = torch.zeros((batch_size, output_size, input_size), dtype=torch.float64)
            xa = x.clone().detach().requires_grad_(True)
            #xa = torch.tensor(x.detach(), requires_grad=True)
            torch.set_grad_enabled(True)  # Apparently when pytorch calls backward, we're in no-grad mode.
            ya = EmgradWrapper.forward(None, xa)  # (batch size, output_size)
            ya.retain_grad()
            # Autograd has to be done on a scalar, so we'll slowly iterate through each element 
            # and reconstruct grad_local but computed with autograd.
            for bidx, batch in enumerate(ya):
                for eidx, element in enumerate(batch):
                    if xa.grad is not None: xa.grad.zero_()
                    element.backward(retain_graph=True)
                    flags = xa.grad.sum(axis=1) != 0
                    flags[bidx] = False
                    assert flags.sum() == 0, "everything outside the batch we are inspecting should be zero"
                    #xa.grad is (batch_size, input_size)
                    gla[bidx, eidx, :] = xa.grad[bidx]

            avg_abs_error = (grad_local - gla).abs().mean()
            avg_abs_gla = gla.abs().mean()
            rel_error = avg_abs_error / avg_abs_gla
            print(f"Avg mag of emgrad errors: {avg_abs_error:.4f}  Avg mag of analytic gradients: {avg_abs_gla:.4f}  Rel: {rel_error:.4f}")
            torch.set_grad_enabled(False)

        # Simple gradient clipping.
        if args.clip:
            grad_local.clip(min=-args.clip, max=args.clip)
            
        # dout/din * dL/dout for all batch elements.
        grad_in = (grad_local.permute(0, 2, 1).type(torch.float32) @ grad_output.unsqueeze(2)).squeeze()
        return grad_in

# Glue
class BlackBoxLayer(nn.Module):
    def __init__(self, black_box_fn):
        super().__init__()
        self.egw = EmgradWrapper()
        EmgradWrapper.black_box_fn = black_box_fn  # This is really gross but whatever

    def forward(self, x):
        return self.egw.apply(x)

# Drop in replacement for BlackBoxLayer that uses autograd.
class AnalyticLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)
    

############################################################
# A few simple black box functions to choose from.
############################################################

def bbf1(x):
    x = 10 * x + 5
    x = x.repeat(1, 2)
    return x

# To be really sure that torch autograd can't operate here, do some operations in numpy.
def bbf1_numpy(x):
    z = x.numpy().copy()
    z = 10 * z + 5
    x = torch.from_numpy(z)
    x = x.repeat(1, 2)
    return x

def bbf2(x):
    x = 10 * x**2 + 5
    x = x.repeat(1, 2)
    return x

def bbf2_numpy(x):
    z = x.numpy().copy()  
    z = 10 * z**2 + 5
    x = torch.from_numpy(z)
    x = x.repeat(1, 2)
    return x

class StatefulBBF:
    def __init__(self):
        self.vals = torch.rand(64)
        
    def __call__(self, x):
        return bbf2_numpy(x) * self.vals

bbf3_numpy = StatefulBBF()
    
def bbf4(x):
    lengths = (x * x).sum(axis=1).sqrt().unsqueeze(1)
    x = x / lengths
    x = x.repeat(1, 2)
    return x

        
############################################################
# Model
############################################################
        
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # Deep net implementation of NWP parameterizations
            nn.Linear(32, 64),  # Potentially high-d input, can be anything
            nn.ReLU6(),
            nn.Linear(64, 32),
            
            # NWP black box.
            BlackBoxLayer(bbf4),
            #AnalyticLayer(bbf4),
            
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
    parser.add_argument("--batch-size", "--bs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("-a", "--analytic-comparison-frac", type=float, default=0.0)
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
    
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)

    model.train(True)
    for step, inputs in enumerate(dl):
        probs = model(inputs)
        loss = ((probs - inputs)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            num_errors = (probs.round() - inputs).abs().sum().item()
            acc = 1.0 - num_errors / inputs.numel()
            print(f"{step=} loss={loss.item():.4f} {acc=:.4f}")

        if acc > 0.99:
            print(f"Reached {acc=:0.4f} in {step} steps.")
            break
        

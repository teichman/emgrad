import math
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
        grad_local = torch.zeros((batch_size, output_size, input_size), dtype=torch.float64, device=x.device)
        for _ in range(args.num_reps):
            for idx in range(input_size):
                x[:, idx] += args.eps
                y2 = EmgradWrapper.forward(None, x)
                x[:, idx] -= args.eps
                grad_local[:, :, idx] += (y2 - y) / args.eps
                
        grad_local /= args.num_reps  # Average across reps.

        # Compute analytic gradient for comparison.
        if torch.rand(1) < args.analytic_comparison_frac:
            gla = torch.zeros((batch_size, output_size, input_size), dtype=torch.float64, device=x.device)
            xa = x.clone().detach().requires_grad_(True)
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
    

############################################################
# A few simple black box functions to choose from.
############################################################

def bbf0(x):
    x = 10 * x + 5
    x = x.repeat(1, 2)
    return x

def bbf1(x):
    x = x**2 + 5
    x = x.repeat(1, 2)
    return x

class StatefulBBF:
    def __init__(self):
        self.vals = torch.rand(64)
    def __call__(self, x):
        return bbf0(x) + bbf1(x) * self.vals
bbf2 = StatefulBBF()
    
def bbf3(x):
    lengths = (x * x).sum(axis=1).sqrt().unsqueeze(1)
    x = x / lengths
    x = x.repeat(1, 2)
    return x

def bbf4(x):
    x = bbf2(x)
    x = torch.where(x > 3, x.log(), x)
    return x

bbfs = [bbf0, bbf1, bbf2, bbf3, bbf4]

# To be really sure that torch autograd can't operate here, do some operations in numpy.
# (Analytic gradient comparisons not possible here, ofc.)
def bbf0_numpy(x):
    z = x.numpy().copy()
    z = 10 * z + 5
    x = torch.from_numpy(z)
    x = x.repeat(1, 2)
    return x

def bbf1_numpy(x):
    z = x.numpy().copy()  
    z = 10 * z**2 + 5
    x = torch.from_numpy(z)
    x = x.repeat(1, 2)
    return x

    
############################################################
# Model
############################################################

# Layer that takes an arbitrary python function and tells pytorch to use emgrad for training.
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Parameterizations, e.g. 
        # Potentially high-d input, e.g. various state of each grid cell.
        # Output of this deep net is a few global parameters
        self.nwp_parameterizations = nn.Sequential(
            nn.Linear(32, 64),  
            nn.ReLU6(),
            nn.Linear(64, 32),
        )

        # NWP numerical solver that blocks gradients.
        self.black_box_layer = BlackBoxLayer(bbfs[args.bbf])
        if args.autograd:
            self.black_box_layer = AnalyticLayer(bbfs[args.bbf])

        # Super-resolution & "style transfer" step.
        self.super_res = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU6(),
            nn.Linear(128, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.ReLU6(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.nwp_parameterizations(x)
        x = self.black_box_layer(x)
        x = self.super_res(x)
        return x

    
############################################################
# Training
############################################################

def train():
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
    
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train(True)
    for step, inputs in enumerate(dl):
        inputs = inputs.to(device)
        probs = model(inputs)
        loss = ((probs - inputs)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            num_errors = (probs.round() - inputs).abs().sum().item()
            acc = 1.0 - num_errors / inputs.numel()
            statstr = f"{step=} loss={loss.item():.4f} {acc=:.4f} logloss={math.log10(loss.item()):.4f}"
            statstr += f" loglr={math.log10(optimizer.param_groups[0]['lr']):.4f} logeps={math.log10(args.eps):.4f}"
            print(statstr)
            args.eps *= args.eps_mult
            args.lr *= args.lr_mult
            optimizer.param_groups[0]['lr'] = args.lr
            assert len(optimizer.param_groups) == 1

        if acc > args.acc_thresh:
            print(f"Reached {acc=:0.4f} in {step} steps.")
            break

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "--bs", type=int, default=60)
    parser.add_argument("--bbf", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--eps-mult", type=float, default=1.0)
    parser.add_argument("--acc-thresh", type=float, default=0.98)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--autograd", action="store_true", help="Use autograd instead of emgrad for comparison.")
    parser.add_argument("-a", "--analytic-comparison-frac", type=float, default=0.0)
    parser.add_argument("--num-reps", type=int, default=1,
                        help="Number of reps of empirical gradient estimation per backprop step.")
    args = parser.parse_args()    

    device = torch.device("cpu")
    if args.gpu is not None:
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using device {device}")
        
    train()

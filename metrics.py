import torch

def compute_accuracy(net, data_loader):
    acc = 0
    total = 0.
    net.eval()
    with torch.no_grad():
        for (X, y, _) in data_loader:
            out, _ = net(X)
            acc += ((out > 0.5) == y).float().sum().item()
            total += len(y)
    return acc/total
import torch


def log_loss(z):
    # stable implementation of logistic loss
    idx = z > 0
    out = torch.zeros_like(z)
    out[idx] = torch.log(1 + torch.exp(-z[idx]))
    out[~idx] = -z[~idx] + torch.log(1 + torch.exp(z[~idx]))
    return out


class LogisticATLoss(torch.nn.Module):
    def __init__(self, alpha=1):
        super(LogisticATLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target, theta, weights, n_classes):
        # reshape input to 1d tensor and compute theta - input
        z1 = theta - input.view(-1)
        n_classes = target.shape[1]

        s = torch.sign(torch.arange(n_classes)[:, None] - target + 0.5)

        print(f"shape of s: {s.shape}, shape of z1: {z1.shape}")

        err = log_loss(s * z1)
        loss = torch.sum(err)
        loss += self.alpha * 0.5 * torch.dot(weights, weights)

        return loss


if __name__ == "__main__":
    # Test
    import numpy as np

    np.random.seed(0)
    torch.manual_seed(0)

    n_classes = 5
    n_samples = 100
    n_features = 1

    X = torch.randn(n_samples, n_features)
    y = torch.zeros(n_samples, 1)

    theta = torch.randn(n_classes, 1)
    weights = torch.randn(n_classes)

    loss = LogisticATLoss(alpha=1)
    out = loss(X, y, theta, weights, n_classes)
    print(out)

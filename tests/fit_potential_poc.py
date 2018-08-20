import pytest
import torch


def test_potential():
    """Fit a potential to observed paths.

    Potential y:
    y = ax
    x_2 = x_1 + eps * dy/dx

    x_1 and x_2 are given. eps=1 for simplicity. Find a.
    """
    a = torch.tensor(1.0, requires_grad=True)
    x_1 = torch.tensor([1.0, 0.0], requires_grad=False)
    x_1.requires_grad_()
    y = a * x_1
    grad_y, = torch.autograd.grad(y, x_1, torch.ones_like(y), create_graph=True)
    print('grad_y', grad_y)

    x_2_target = torch.tensor([1.5, 1.0])
    x_2 = x_1 + grad_y
    loss = (x_2_target - x_2).pow(2).mean()
    grad_a, = torch.autograd.grad(loss, a)
    print('loss', loss, loss.grad)
    print('a', a, a.grad, grad_a)

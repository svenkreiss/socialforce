import math
import pytest
import torch
import socialforce

import numpy as np


def test_rab():
    state = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ], requires_grad=True)
    V = socialforce.PedPedPotential(0.4)
    assert V.r_ab(state).tolist() == [[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]]


def test_f_pedped():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ], requires_grad=True)
    s = socialforce.Simulator(initial_state)
    force_at_unit_distance = 0.25  # confirmed below
    assert s.f_pedped().detach().numpy() == pytest.approx(np.array([
        [-force_at_unit_distance, 0.0],
        [force_at_unit_distance, 0.0],
    ]), abs=0.05)


def test_b_zero_vel():
    r_ab = torch.tensor([[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]])
    speeds = torch.tensor([0.0, 0.0])
    desired_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    V = socialforce.PedPedPotential(0.4)
    assert V.b(r_ab, speeds, desired_directions).tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


def test_torch_potential_gradient():
    state = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ], requires_grad=True)
    v0 = torch.Tensor([2.1])
    sigma = torch.Tensor([0.3])

    r = state[:, 0:2]
    r_a = r.unsqueeze(1)
    r_b = r.unsqueeze(0).detach()  # !!!!!!!!!! otherwise gradient of b will accumulate into a
    r_ab = r_a - r_b
    r_ab_norm = torch.norm(r_ab, dim=-1)
    print(r_ab_norm)

    pedped_potential = v0 * torch.exp(-r_ab_norm / sigma)
    diag = torch.eye(pedped_potential.shape[0], dtype=torch.uint8)
    pedped_potential[diag] = 0.0
    # pedped_potential = torch.sum(pedped_potential, dim=1)
    print('value', pedped_potential)
    gradients = torch.ones_like(pedped_potential)
    pedped_potential.backward(gradients)
    print(state.grad)

    analytic_abs_grad_value = 2.1 * math.exp(-1.0/0.3) * 1.0/0.3
    print(analytic_abs_grad_value)
    assert state.grad[0][0] == analytic_abs_grad_value
    assert state.grad[1][0] == -analytic_abs_grad_value

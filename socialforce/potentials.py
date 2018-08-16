"""Interaction potentials."""

import torch

from . import stateutils


class PedPedPotential(object):
    """Ped-ped interaction potential.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    def __init__(self, delta_t, v0=2.1, sigma=0.3):
        self.delta_t = delta_t
        self.v0 = v0
        self.sigma = sigma

    def b(self, r_ab, speeds, desired_directions):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        in_sqrt = (
            torch.norm(r_ab, dim=-1) +
            torch.norm(r_ab - self.delta_t * speeds_b_abc * e_b, dim=-1)
        )**2 - (self.delta_t * speeds_b)**2
        in_sqrt[torch.eye(in_sqrt.shape[0], dtype=torch.uint8)] = 0.0

        return 0.5 * torch.sqrt(in_sqrt)

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        return self.v0 * torch.exp(-self.b(r_ab, speeds, desired_directions) / self.sigma)

    @staticmethod
    def r_ab(state):
        """Construct r_ab using broadcasting."""
        r = state[:, 0:2]
        r_a0 = r.unsqueeze(1)
        r_0b = r.unsqueeze(0)
        return r_a0 - r_0b

    def __call__(self, state):
        speeds = stateutils.speeds(state)
        return self.value_r_ab(self.r_ab(state), speeds, stateutils.desired_directions(state))

    def grad_r_ab_finite_difference(self, state, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state[:, 0:2])
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        dvdx[torch.eye(dvdx.shape[0], dtype=torch.uint8)] = 0.0
        dvdy[torch.eye(dvdx.shape[0], dtype=torch.uint8)] = 0.0

        return torch.stack((dvdx, dvdy), dim=-1)

    def grad_r_ab(self, state):
        """Compute gradient wrt r_ab using autograd."""
        r_ab = self.r_ab(state).clone().detach().requires_grad_()
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        v.backward(torch.ones_like(v), retain_graph=True)
        r_ab_grad = torch.autograd.grad(v, r_ab, torch.ones_like(v))
        return r_ab.grad
        # v, r_ab_grad = PartialGradient().apply(v, r_ab)
        # return r_ab_grad


class PartialGradient(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, v, r_ab):
        return v, r_ab.grad.detach()

    @staticmethod
    def backward(ctx, grad_output, _):
        # do nothing for now
        return grad_output.clone(), None


class PedPedPotentialMLP(object):
    """Ped-ped interaction potential."""

    def __init__(self, delta_t, parameters):
        self.delta_t = delta_t

        lin1 = torch.nn.Linear(1, 5)
        lin1.weight = torch.nn.Parameter(parameters[0][0])
        lin1.bias = torch.nn.Parameter(parameters[0][1])
        lin2 = torch.nn.Linear(5, 1)
        lin2.weight = torch.nn.Parameter(parameters[1][0])
        lin2.bias = torch.nn.Parameter(parameters[1][1])
        self.mlp = torch.nn.Sequential(lin1, torch.nn.Tanh(), lin2)

    def b(self, r_ab, speeds, desired_directions):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        in_sqrt = (
            torch.norm(r_ab, dim=-1) +
            torch.norm(r_ab - self.delta_t * speeds_b_abc * e_b, dim=-1)
        )**2 - (self.delta_t * speeds_b)**2
        in_sqrt[torch.eye(in_sqrt.shape[0], dtype=torch.uint8)] = 0.0

        return 0.5 * torch.sqrt(in_sqrt)

    def value_b(self, b):
        """Calculate value given b."""
        # modified_b = 3.0 - b
        # modified_b[torch.eye(modified_b.shape[0], dtype=torch.uint8)] = 0.0
        v = self.mlp(-b.view(-1, 1)).view(b.shape)
        v[b > 3.0] = 0.0
        return v

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
        # return self.v0 * torch.exp(-b / self.sigma)
        return self.value_b(b)

    @staticmethod
    def r_ab(state):
        """Construct r_ab using broadcasting."""
        r = state[:, 0:2]
        r_a0 = r.unsqueeze(1)
        r_0b = r.unsqueeze(0)
        return r_a0 - r_0b

    def __call__(self, state):
        speeds = stateutils.speeds(state)
        return self.value_r_ab(self.r_ab(state), speeds, stateutils.desired_directions(state))

    def grad_r_ab_finite_difference(self, state, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state[:, 0:2])
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        dvdx[torch.eye(dvdx.shape[0], dtype=torch.uint8)] = 0.0
        dvdy[torch.eye(dvdx.shape[0], dtype=torch.uint8)] = 0.0

        return torch.stack((dvdx, dvdy), dim=-1)

    def grad_r_ab(self, state):
        """Compute gradient wrt r_ab using autograd."""
        r_ab = self.r_ab(state).clone().detach().requires_grad_()
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        v.backward(torch.ones_like(v), retain_graph=True)
        # r_ab_grad = torch.autograd.grad(v, r_ab, torch.ones_like(v))
        return r_ab.grad
        # v, r_ab_grad = PartialGradient().apply(v, r_ab)
        # return r_ab_grad


class PedSpacePotential(object):
    """Pedestrian-space interaction potential.

    space is a list of numpy arrays containing points of boundaries.

    u0 is in m^2 / s^2.
    r is in m
    """

    def __init__(self, space, u0=10, r=0.2):
        self.space = space or []
        self.u0 = u0
        self.r = r

    def value_r_aB(self, r_aB):
        """Compute value parametrized with r_aB."""
        return self.u0 * torch.exp(-1.0 * torch.norm(r_aB, dim=-1) / self.r)

    def r_aB(self, state):
        """r_aB"""
        if not self.space:
            return torch.zeros((state.shape[0], 0, 2))

        r_a = state[:, 0:2].unsqueeze(1)
        closest_i = [
            torch.argmin(torch.norm(r_a - B.unsqueeze(0), dim=-1), dim=1)
            for B in self.space
        ]
        closest_points = torch.transpose(
            torch.stack([B[i] for B, i in zip(self.space, closest_i)]),
            0, 1)  # index order: pedestrian, boundary, coordinates
        return r_a - closest_points

    def __call__(self, state):
        return self.value_r_aB(self.r_aB(state))

    def grad_r_aB(self, state, delta=1e-3):
        """Compute gradient wrt r_aB using finite difference differentiation."""
        r_aB = self.r_aB(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_aB(r_aB)
        dvdx = (self.value_r_aB(r_aB + dx) - v) / delta
        dvdy = (self.value_r_aB(r_aB + dy) - v) / delta

        return torch.stack((dvdx, dvdy), dim=-1)

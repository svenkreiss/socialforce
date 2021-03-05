"""Interaction potentials."""

import torch

from . import stateutils


class PedPedPotential(torch.nn.Module):
    """Ped-ped interaction potential based on distance b.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    def __init__(self, v0=2.1, sigma=0.3):
        super().__init__()
        self.v0 = v0
        self.sigma = sigma

    def b(self, r_ab, speeds, desired_directions, delta_t):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        in_sqrt = (
            self.norm_r_ab(r_ab)
            + self.norm_r_ab(r_ab - delta_t * speeds_b_abc * e_b)
        )**2 - (delta_t * speeds_b)**2

        # torch.diagonal(in_sqrt)[:] = 0.0  # protect forward pass
        in_sqrt = torch.clamp(in_sqrt, min=1e-8)
        out = 0.5 * torch.sqrt(in_sqrt)
        # torch.diagonal(out)[:] = 0.0  # protect backward pass

        return out

    def value_b(self, b):
        """Value of potential parametrized with b."""
        return self.v0 * torch.exp(-b / self.sigma)

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions, delta_t)
        return self.value_b(b)

    @staticmethod
    def r_ab(state):
        """Construct r_ab using broadcasting."""
        r = state[:, 0:2]
        r_a0 = r.unsqueeze(1)
        r_0b = r.unsqueeze(0)
        r_ab = r_a0 - r_0b
        torch.diagonal(r_ab)[:] = 0.0  # detach diagonal gradients
        return r_ab

    def forward(self, state, *, delta_t):
        speeds = stateutils.speeds(state).detach()
        desired_directions = stateutils.desired_directions(state).detach()
        return self.value_r_ab(self.r_ab(state), speeds, desired_directions, delta_t)

    def grad_r_ab_finite_difference(self, state, delta_t, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state[:, 0:2])
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = torch.tensor([[[delta, 0.0]]])
        dy = torch.tensor([[[0.0, delta]]])

        v = self.value_r_ab(r_ab, speeds, desired_directions, delta_t)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions, delta_t) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions, delta_t) - v) / delta

        # remove gradients from self-intereactions
        torch.diagonal(dvdx)[:] = 0.0
        torch.diagonal(dvdy)[:] = 0.0

        return torch.stack((dvdx, dvdy), dim=-1)

    def grad_r_ab(self, state, delta_t):
        """Compute gradient wrt r_ab using autograd."""
        speeds = stateutils.speeds(state).detach()
        desired_directions = stateutils.desired_directions(state).detach()

        def compute(r_ab):
            return self.value_r_ab(r_ab, speeds, desired_directions, delta_t)

        r_ab = self.r_ab(state)
        r_ab = torch.clamp(r_ab, -100, 100)  # to avoid infinities / nans
        with torch.enable_grad():
            vector = torch.ones(r_ab.shape[0:2], requires_grad=False)
            _, r_ab_grad = torch.autograd.functional.vjp(
                compute, r_ab, vector,
                create_graph=True, strict=True)

        return r_ab_grad

    @staticmethod
    def norm_r_ab(r_ab):
        """Norm of r_ab.

        Special treatment of diagonal terms for backpropagation.

        Without this treatment, backpropagating through a norm of a
        zero vector gives nan gradients.
        """
        out = torch.norm(r_ab, dim=-1, keepdim=False)
        torch.diagonal(out)[:] = 0.0
        return out


class PedPedPotentialMLP(PedPedPotential):
    """Ped-ped interaction potential."""

    def __init__(self, *, hidden_units=5):
        super().__init__()

        lin1 = torch.nn.Linear(1, hidden_units)
        lin2 = torch.nn.Linear(hidden_units, 1, bias=False)

        # initialize
        torch.nn.init.normal_(lin1.weight, std=0.03)
        torch.nn.init.normal_(lin1.bias, std=0.03)
        torch.nn.init.normal_(lin2.weight, std=0.03)

        self.mlp = torch.nn.Sequential(lin1, torch.nn.Tanh(), lin2)

    def value_b(self, b):
        """Calculate value given b."""
        b = torch.clamp(b, max=100.0)
        return self.mlp(b.view(-1, 1)).view(b.shape)


class PedSpacePotential(torch.nn.Module):
    """Pedestrian-space interaction potential.

    space is a list of numpy arrays containing points of boundaries.

    u0 is in m^2 / s^2.
    r is in m
    """

    def __init__(self, space, u0=10, r=0.2):
        super().__init__()
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

    def forward(self, state):
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

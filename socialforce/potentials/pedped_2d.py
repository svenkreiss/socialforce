"""Interaction potentials."""

import torch

from .. import stateutils
from ..field_of_view import FieldOfView
from .pedped_1d import PedPedPotential


class PedPedPotential2D(torch.nn.Module):
    """Ped-ped interaction potential based on distance b and relative angle.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    b = PedPedPotential.b
    value_b = PedPedPotential.value_b
    r_ab = PedPedPotential.r_ab
    norm_r_ab = staticmethod(PedPedPotential.norm_r_ab)
    grad_r_ab = PedPedPotential.grad_r_ab
    forward = PedPedPotential.forward

    def __init__(self, v0=2.1, sigma=0.3, field_of_view: FieldOfView = None):
        super().__init__()
        self.v0 = v0
        self.sigma = sigma
        self.field_of_view = field_of_view

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions, delta_t)
        value = self.value_b(b)
        if self.field_of_view is not None:
            # r_ab is grad_r_ab in the paper
            w = self.field_of_view(desired_directions, r_ab).detach()
            value = value * w
        return value


class PedPedPotentialMLP2D(PedPedPotential2D):
    """Ped-ped interaction potential."""

    def __init__(self, *, hidden_units=5):
        super().__init__()

        lin1 = torch.nn.Linear(3, hidden_units)
        lin2 = torch.nn.Linear(hidden_units, 1, bias=False)

        # initialize
        torch.nn.init.normal_(lin1.weight, std=0.03)
        torch.nn.init.normal_(lin1.bias, std=0.03)
        torch.nn.init.normal_(lin2.weight, std=0.03)

        self.mlp = torch.nn.Sequential(lin1, torch.nn.Tanh(), lin2)

    def value_r_ab(self, r_ab, speeds, desired_directions, delta_t):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions, delta_t)
        b = torch.clamp(b, max=100.0)
        unit_r_ab = r_ab / self.norm_r_ab(r_ab)
        cos = torch.dot(unit_r_ab, desired_directions)
        sin = torch.norm(torch.cross(unit_r_ab, desired_directions), dim=-1)

        input_vector = torch.stack((b, cos, sin), dim=-1)
        return self.mlp(input_vector.view(-1, 3)).view(b.shape)

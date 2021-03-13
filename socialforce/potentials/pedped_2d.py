"""Interaction potentials."""

import math
import torch

from .pedped_1d import PedPedPotential, PedPedPotentialMLP


class PedPedPotential2D(torch.nn.Module):
    """Ped-ped interaction potential based on distance b and relative angle.

    v0 is in m^2 / s^2.
    sigma is in m.
    """
    delta_t_step = PedPedPotential.delta_t_step

    b = PedPedPotential.b
    value_b = PedPedPotential.value_b
    r_ab = staticmethod(PedPedPotential.r_ab)
    norm_r_ab = staticmethod(PedPedPotential.norm_r_ab)
    grad_r_ab = PedPedPotential.grad_r_ab
    grad_r_ab_ = PedPedPotential.grad_r_ab_
    forward = PedPedPotential.forward

    def __init__(self, v0=2.1, sigma=0.3, asymmetry=0.0):
        super().__init__()

        self.v0 = v0
        self.sigma = sigma
        self.asymmetry = asymmetry
        self.register_buffer('rot90', torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))

    @staticmethod
    def parallel_d(r_ab, desired_directions):
        parallel_d = torch.einsum('abj,bj->ab', r_ab, desired_directions)
        torch.diagonal(parallel_d)[:] = 0.0
        return parallel_d

    def perpendicular_d(self, r_ab, desired_directions):
        desired_directions_p = torch.matmul(desired_directions, self.rot90)
        perpendicular_d = torch.einsum('abj,bj->ab', r_ab, desired_directions_p)
        torch.diagonal(perpendicular_d)[:] = 0.0
        return perpendicular_d

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
        value = self.value_b(b)
        if self.asymmetry != 0.0:
            perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
            factor = torch.clamp_min(1.0 + self.asymmetry * perpendicular_d, 0.0)
            value = factor * value
        return value


class PedPedPotentialDiamond(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, v0=2.1, sigma=0.3, asymmetry=0.0):
        super().__init__()

        self.v0 = v0
        self.sigma = sigma
        self.asymmetry = asymmetry

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        assert speeds.requires_grad is False
        assert desired_directions.requires_grad is False

        parallel_d = self.parallel_d(r_ab, desired_directions)
        perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
        l1 = torch.linalg.norm(
            torch.stack((parallel_d, perpendicular_d), dim=-1),
            ord=1, dim=-1,
        )
        return self.v0 * torch.clamp_min(1.0 - 0.5 / self.sigma * l1, 0.0)


class PedPedPotentialMLP1p1D(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, *, hidden_units=5):
        super().__init__()

        self.pedped_b = PedPedPotentialMLP(hidden_units=hidden_units)

        perpendicular_lin1 = torch.nn.Linear(1, hidden_units)
        perpendicular_lin2 = torch.nn.Linear(hidden_units, 1)
        self.mlp_perpendicular = torch.nn.Sequential(
            perpendicular_lin1, torch.nn.Softplus(),
            perpendicular_lin2, torch.nn.Softplus(),
        )

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        assert speeds.requires_grad is False
        assert desired_directions.requires_grad is False
        out_b = self.pedped_b.value_r_ab(r_ab, speeds, desired_directions)

        perpendicular_d = self.perpendicular_d(r_ab, desired_directions)
        out_perpendicular = self.mlp_perpendicular(
            perpendicular_d.reshape(-1, 1)
        ).view(r_ab[:, :, 0].shape)

        out = torch.mul(out_b, out_perpendicular)
        return out


class PedPedPotentialMLP2D(PedPedPotential2D):
    """Ped-ped interaction potential."""
    def __init__(self, *, hidden_units=16, n_fourier_features=None, fourier_scale=1.0):
        super().__init__()

        input_features = 3

        if n_fourier_features:
            fourier_featurizer = torch.randn((input_features, n_fourier_features // 2))
            self.register_buffer('fourier_featurizer', fourier_featurizer)
            input_features = n_fourier_features
        else:
            self.fourier_featurizer = None
        self.fourier_scale = fourier_scale

        lin1 = torch.nn.Linear(input_features, hidden_units)
        lin2 = torch.nn.Linear(hidden_units, hidden_units)
        lin3 = torch.nn.Linear(hidden_units, 1)

        self.mlp = torch.nn.Sequential(
            lin1, torch.nn.Softplus(),
            lin2, torch.nn.Softplus(),
            lin3, torch.nn.Softplus(),
        )

    def input_features(self, r_ab, speeds, desired_directions):
        input_vector = torch.stack((
            self.b(r_ab, speeds, desired_directions),
            self.perpendicular_d(r_ab, desired_directions),
            self.parallel_d(r_ab, desired_directions),
        ), dim=-1)

        if self.fourier_featurizer is not None:
            input_vector = self.fourier_features(input_vector)

        return input_vector

    def fourier_features(self, input_vector):
        input_vector = torch.tanh(input_vector / self.fourier_scale) * 2.0 * math.pi
        ff = torch.matmul(input_vector, self.fourier_featurizer)
        ff = torch.cat((torch.sin(ff), torch.cos(ff)), dim=-1)
        return ff

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        input_vector = self.input_features(r_ab, speeds, desired_directions)
        flattened = input_vector.view(-1, input_vector.shape[-1])
        out = self.mlp(flattened).view(r_ab[:, :, 0].shape)
        out = out.clone()
        torch.diagonal(out)[:] = 0.0
        return out

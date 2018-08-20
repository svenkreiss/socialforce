"""Interaction potentials."""

import torch

from . import stateutils


class PedPedPotential(object):
    """Ped-ped interaction potential.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    def __init__(self, delta_t, v0=None, sigma=0.3):
        self.delta_t = delta_t
        self.v0 = v0
        if self.v0 is None:
            self.v0 = torch.tensor(2.1, requires_grad=True)
        self.sigma = sigma

    def b(self, r_ab, speeds, desired_directions):
        """Calculate b."""
        speeds_b = speeds.unsqueeze(0)
        speeds_b_abc = speeds_b.unsqueeze(2)  # abc = alpha, beta, coordinates
        e_b = desired_directions.unsqueeze(0)

        in_sqrt = (
            self.norm_r_ab(r_ab) +
            self.norm_r_ab(r_ab - self.delta_t * speeds_b_abc * e_b)
        )**2 - (self.delta_t * speeds_b)**2

        # in_sqrt[torch.eye(in_sqrt.shape[0], dtype=torch.uint8)] = 0.0  # protect forward pass
        in_sqrt = torch.clamp(in_sqrt, min=0.0000001)
        out = 0.5 * torch.sqrt(in_sqrt)
        # out[torch.eye(out.shape[0], dtype=torch.uint8)] = 0.0  # protect backward pass

        return 0.5 * out

    def value_b(self, b):
        """Value of potential parametrized with b."""
        return self.v0 * torch.exp(-b / self.sigma)

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitely parametrized with r_ab."""
        b = self.b(r_ab, speeds, desired_directions)
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
        r_ab = torch.clamp(r_ab, -100, 100)  # to avoid infinities / nans
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        r_ab_grad, = torch.autograd.grad(v, r_ab, torch.ones_like(v), create_graph=True)
        return r_ab_grad

    def norm_r_ab(self, r_ab):
        """Norm of r_ab.

        Special treatment of diagonal terms for backpropagation.

        Without this treatment, backpropagating through a norm of a
        zero vector gives nan gradients.
        """
        out = torch.norm(r_ab, dim=-1, keepdim=True).clone()
        out[torch.eye(out.shape[0], dtype=torch.uint8)] = 0.0
        return out


class PedPedPotentialMLP(PedPedPotential):
    """Ped-ped interaction potential."""

    def __init__(self, delta_t, hidden_units=5):
        super(PedPedPotentialMLP, self).__init__(delta_t)
        self.hidden_units = hidden_units

        self.lin1 = torch.nn.Linear(1, hidden_units)
        self.lin2 = torch.nn.Linear(hidden_units, 1, bias=False)

        # initialize
        torch.nn.init.normal_(self.lin1.weight, std=0.03)
        torch.nn.init.normal_(self.lin1.bias, std=0.03)
        torch.nn.init.normal_(self.lin2.weight, std=0.03)

        self.mlp = torch.nn.Sequential(self.lin1, torch.nn.Tanh(), self.lin2)

    def parameters(self):
        return [
            self.lin1.weight,
            self.lin1.bias,
            self.lin2.weight,
        ]

    def get_parameters(self):
        """Returns a single 1D array of parameters."""
        return torch.cat((
            self.lin1.weight.view(-1),
            self.lin1.bias,
            self.lin2.weight.view(-1),
        ))

    def set_parameters(self, parameters_1d):
        """Set parameters of the MLP from the given 1D parameter tensor."""
        with torch.no_grad():
            i = 0
            for p in (self.lin1.weight, self.lin1.bias, self.lin2.weight):
                n = p.nelement()
                p[:] = parameters_1d[i:i + n].view(p.shape)
                i += n

    def value_b(self, b):
        """Calculate value given b."""
        # modified_b = 3.0 - b
        # modified_b[torch.eye(modified_b.shape[0], dtype=torch.uint8)] = 0.0
        # print('b', b)
        b = torch.clamp(b, max=10.0)
        v = self.mlp(b.view(-1, 1)).view(b.shape)
        # v[torch.eye(v.shape[0], dtype=torch.uint8)] = 0.0
        # print(b, v)
        # v[b > 3.0] = 0.0
        return v


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

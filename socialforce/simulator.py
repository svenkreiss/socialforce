# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import torch

from .potentials import PedPedPotential
from .fieldofview import FieldOfView
from . import stateutils

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, [d_x, d_y], [tau]),
    which are the coordinates for position, velocity and destination.
    destination and tau are optional in this vector.

    ped_space is an instance of PedSpacePotential.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
    """
    def __init__(self, initial_state, *,
                 ped_space=None, ped_ped=None, delta_t=0.4, tau=0.5,
                 oversampling=10, dtype=None):
        if dtype is None:
            dtype = torch.double
        if not isinstance(initial_state, torch.Tensor):
            initial_state = torch.tensor(initial_state, dtype=dtype)

        self.state = initial_state
        self.desired_speeds = stateutils.speeds(initial_state)
        self.max_speeds = MAX_SPEED_MULTIPLIER * self.desired_speeds

        self.delta_t = delta_t / oversampling
        self.oversampling = oversampling

        if self.state.shape[1] == 4:
            # destinations and tau not given
            no_destinations = torch.full((self.state.shape[0], 2), float('nan'),
                                         dtype=self.state.dtype)
            self.state = torch.cat((self.state, no_destinations), dim=-1)
        if self.state.shape[1] == 5:
            # destinations not given but tau is given
            no_destinations = torch.full((self.state.shape[0], 2), float('nan'),
                                         dtype=self.state.dtype)
            self.state = torch.cat(
                (self.state[:, 0:4], no_destinations, self.state[:, 4:]), dim=-1)
        if self.state.shape[1] == 6:
            # tau not given
            if not hasattr(tau, 'shape'):
                tau = tau * torch.ones(self.state.shape[0], dtype=self.state.dtype)
            self.state = torch.cat((self.state, tau.unsqueeze(-1)), dim=-1)

        # potentials
        self.V = ped_ped or PedPedPotential()
        self.U = ped_space

        # field of view
        self.w = FieldOfView()

    def f_ab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(self.state, self.delta_t)

    def f_aB(self):
        """Compute f_aB."""
        if self.U is None:
            return None
        return -1.0 * self.U.grad_r_aB(self.state)

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = torch.norm(desired_velocity, dim=-1)
        # factor = torch.minimum(1.0, self.max_speeds / desired_speeds)
        factor = torch.clamp(self.max_speeds / desired_speeds, max=1.0)
        return desired_velocity * factor.unsqueeze(-1)

    def step(self):
        """Do oversampling steps."""
        for _ in range(self.oversampling):
            self._step()

        return self

    def _step(self):
        """Do one step in the simulation and update the state in place."""
        self.state = self.state.clone().detach()

        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = 1.0 / tau * (self.desired_speeds.unsqueeze(-1) * e - vel)

        # repulsive terms between pedestrians
        F_ab = None
        f_ab = self.f_ab()
        if f_ab is not None:
            w = self.w(e, -f_ab).unsqueeze(-1)
            F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB()

        # social force
        F = F0
        if F_ab is not None:
            F += torch.sum(F_ab, dim=1)
        if F_aB is not None:
            F += torch.sum(F_aB, dim=1)

        # before applying updates to state, make a copy of it
        self.state = self.state.clone().detach()

        # desired velocity
        w = self.state[:, 2:4] + self.delta_t * F
        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * self.delta_t
        self.state[:, 2:4] = v

        return self

    def run(self, n_steps, *, detach=True):
        return torch.stack([
            self.state.clone().detach()
        ] + [
            self.step().state.clone().detach()
            if detach else self.step().state.clone()
            for _ in range(n_steps)
        ])

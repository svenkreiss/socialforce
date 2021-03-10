import math
import numpy as np
import torch

from . import potentials
from .simulator import Simulator


class Circle:
    def __init__(self, ped_ped=None):
        self.ped_ped = ped_ped or potentials.PedPedPotential(2.1)

    def generate(self, n):
        torch.manual_seed(42)
        np.random.seed(42)

        # ped0 always left to right
        ped0 = np.array([-5.0, 0.0, 1.0, 0.0, 5.0, 0.0])

        generator_initial_states = []
        for theta in np.random.rand(n) * 2.0 * math.pi:
            # ped1 at a random angle with +/-20% speed variation
            c, s = np.cos(theta), np.sin(theta)
            r = np.array([[c, -s], [s, c]])
            ped1 = np.concatenate((
                np.matmul(r, ped0[0:2]),
                np.matmul(r, ped0[2:4]) * (0.8 + np.random.rand(1) * 0.4),
                np.matmul(r, ped0[4:6]),
            ))
            generator_initial_states.append(
                np.stack((ped0, ped1))
            )

        with torch.no_grad():
            return [
                Simulator(initial_state, ped_ped=self.ped_ped).run(21)
                for initial_state in generator_initial_states
            ]


class ParallelOvertake:
    def __init__(self, ped_ped=None):
        self.ped_ped = ped_ped or potentials.PedPedPotential(2.1)

    def generate(self, n):
        torch.manual_seed(42)
        np.random.seed(42)

        # ped0 always left to right
        ped0 = [-5.0, 0.0, 1.0, 0.0, 5.0, 0.0]

        generator_initial_states = []
        for b in -0.3 + 0.6 * np.random.rand(n):
            # with 20% speed variation
            speed = 1.3 + 0.2 * np.random.rand(1)[0]
            ped1 = [-7.0, b, speed, 0.0, 7.0, b]

            state = np.array([ped0, ped1])
            generator_initial_states.append(state)

        with torch.no_grad():
            return [
                Simulator(initial_state, ped_ped=self.ped_ped).run(21)
                for initial_state in generator_initial_states
            ]

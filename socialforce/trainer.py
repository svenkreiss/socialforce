import logging
import random

import torch

from .potentials import PedPedPotential

LOG = logging.getLogger(__name__)


class Trainer:
    """Trainer

    true_experience: list of state tuples (each state is a torch.Tensor)
    """

    def __init__(self, simulator_factory, optimizer, true_experience, *,
                 batch_size=1, loss=None):
        self.simulator_factory = simulator_factory
        self.optimizer = optimizer
        self.true_experience = true_experience

        self.batch_size = batch_size
        self.loss = loss
        if loss is None:
            self.loss = torch.nn.SmoothL1Loss(beta=0.05)

    @staticmethod
    def scenes_to_experience(scenes, radius=2.0):
        experience = [
            (scene[0], state1, state2)
            for scene in scenes
            for state1, state2 in zip(scene[:-1], scene[1:])
        ]
        n_total = len(experience)

        def keep(state):
            small_distance = PedPedPotential.norm_r_ab(PedPedPotential.r_ab(state)) < radius
            torch.diagonal(small_distance)[:] = False
            return torch.any(small_distance, dim=-1)

        keep_pedestrians = [keep(state) for _, state, __ in experience]
        experience = [
            (initial[k], state1[k], state2[k])
            for k, (initial, state1, state2) in zip(keep_pedestrians, experience)
            if torch.any(k)
        ]

        LOG.info('from %d scenes, extracted %d experiences, filtered to %d',
                 len(scenes), n_total, len(experience))
        return experience

    def sim_step(self, initial_state, step_state):
        s = self.simulator_factory(initial_state)
        return s(step_state)

    def epoch(self):
        # data = np.random.(self.true_experience)
        random.shuffle(self.true_experience)

        n_batches = 0
        epoch_loss = 0.0

        for i in range(0, len(self.true_experience), self.batch_size):
            data = self.true_experience[i:i + self.batch_size]

            loss = 0.0
            for e in data:
                Y = e[2]
                X = self.sim_step(e[0], e[1])
                loss = loss + self.loss(X[:, :2], Y[:, :2])

            epoch_loss += float(loss.item())
            n_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= n_batches
        return epoch_loss

    def loop(self, n_epochs, log_interval=10):
        for i in range(n_epochs):
            loss = self.epoch()
            if (i + 1) % log_interval == 0:
                print(f'epoch {i + 1}: {loss}')

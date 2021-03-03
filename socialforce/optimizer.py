import random

import torch


class Optimizer(object):
    """Optimizer

    true_experience: list of state tuples (each state is a torch.Tensor)
    """

    def __init__(self, simulator_factory, parameters, true_experience, *,
                 batch_size=1, lr=0.3, loss=None):
        self.simulator_factory = simulator_factory
        self.parameters = parameters
        self.true_experience = true_experience

        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        if loss is None:
            self.loss = torch.nn.SmoothL1Loss(beta=0.05)

    @staticmethod
    def scenes_to_experience(scenes):
        return [
            (state1, state2)
            for scene in scenes
            for state1, state2 in zip(scene[:-1], scene[1:])
        ]

    def sim_step(self, initial_state):
        s = self.simulator_factory(initial_state)
        return s.step()

    def epoch(self):
        # data = np.random.(self.true_experience)
        random.shuffle(self.true_experience)

        n_batches = 0
        epoch_loss = 0.0

        for i in range(0, len(self.true_experience), self.batch_size):
            data = self.true_experience[i:i + self.batch_size]
            Y = torch.stack([e[1] for e in data])
            X = torch.stack([self.sim_step(e[0]) for e in data])

            # loss = ((Y[:, :, :2] - X[:, :, :2]) * 10.0).pow(2).mean()
            loss = self.loss(X[:, :, :2], Y[:, :, :2])

            epoch_loss += float(loss.item())
            n_batches += 1

            # manual SGD-type optimization
            p_grads = torch.autograd.grad(loss, self.parameters)
            with torch.no_grad():
                for p, p_grad in zip(self.parameters, p_grads):
                    p -= self.lr * p_grad

        epoch_loss /= n_batches
        return epoch_loss

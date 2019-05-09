import argparse
import numpy as np
import torch

import trajnettools

from .optimizer import Optimizer
from .potentials import PedPedPotentialMLP
from .simulator import Simulator


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='input data')
    return parser.parse_args()


def generate_experience(args):
    experience = []
    for _, scene in trajnettools.load_all(args.data):
        # scene is of the form [frame, ped, xy]
        positions = scene[1:-1]
        velocities = (scene[2:] - scene[:-2]) / 0.8
        goals = positions + 10.0 * velocities
        states = np.concatenate((positions, velocities, goals), axis=-1)

        # select states with the same pedestrians
        for s1, s2 in zip(states[:-1], states[1:]):
            active_mask = np.logical_and(s1[:, 0] == s1[:, 0],
                                         s1[:, 2] == s1[:, 2])
            if not np.any(active_mask) or np.any(np.isnan(s2[active_mask])):
                continue
            experience.append((
                torch.from_numpy(s1[active_mask].copy()).float(),
                torch.from_numpy(s2[active_mask].copy()).float(),
            ))

    print('experience', len(experience))

    print('============DONE WITH GENERATION===============')

    return experience


def main():
    args = cli()
    experience = generate_experience(args)

    V = PedPedPotentialMLP(delta_t=0.4)
    # initial_parameters = V.get_parameters().clone().detach().numpy()

    def simulator_factory(initial_state):
        return Simulator(initial_state, ped_ped=V, delta_t=0.4)

    opt = Optimizer(simulator_factory, V.parameters(), experience, lr=0.01, batch_size=1)
    for i in range(100):
        loss = opt.epoch()
        print('epoch {}: {}'.format(i, loss))

    # make plots of result
    # visualize('docs/mlp_circle_n{}_'.format(n), V, initial_parameters,
    #           V.get_parameters().clone(), V_gen=generator_ped_ped)


if __name__ == '__main__':
    main()

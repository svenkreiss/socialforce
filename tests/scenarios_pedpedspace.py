from contextlib import contextmanager
import numpy as np
import pytest
import socialforce
import torch


@contextmanager
def visualize(states, space, output_filename):
    import matplotlib.pyplot as plt

    states = states.detach().numpy()
    space = [s.numpy() for s in space]

    print('')
    with socialforce.show.animation(
            len(states),
            output_filename,
            writer='imagemagick') as context:
        ax = context['ax']
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        for s in space:
            ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        actors = []
        for ped in range(states.shape[1]):
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(states[0, ped, 0:2], radius=radius,
                           facecolor='black' if states[0, ped, 4] > 0 else 'white',
                           edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context['update_function'] = update


@pytest.mark.plot
def test_separator():
    initial_state = torch.tensor([
        [-10.0, -0.0, 1.0, 0.0, 10.0, 0.0],
    ])
    space = [
        torch.tensor([(i, i) for i in np.linspace(-1, 4.0, 500)]),
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))
    states = torch.stack([s.step().state.clone() for _ in range(80)])

    # visualize
    with visualize(states, space, 'docs/separator.gif') as ax:
        ax.set_xlim(-10, 10)


@pytest.mark.plot
def test_gate():
    initial_state = torch.tensor([
        [-9.0, -0.0, 1.0, 0.0, 10.0, 0.0],
        [-10.0, -1.5, 1.0, 0.0, 10.0, 0.0],
        [-10.0, -2.0, 1.0, 0.0, 10.0, 0.0],
        [-10.0, -2.5, 1.0, 0.0, 10.0, 0.0],
        [-10.0, -3.0, 1.0, 0.0, 10.0, 0.0],
        [10.0, 1.0, -1.0, 0.0, -10.0, 0.0],
        [10.0, 2.0, -1.0, 0.0, -10.0, 0.0],
        [10.0, 3.0, -1.0, 0.0, -10.0, 0.0],
        [10.0, 4.0, -1.0, 0.0, -10.0, 0.0],
        [10.0, 5.0, -1.0, 0.0, -10.0, 0.0],
    ])
    space = [
        torch.tensor([(0.0, y) for y in np.linspace(-10, -0.7, 1000)]),
        torch.tensor([(0.0, y) for y in np.linspace(0.7, 10, 1000)]),
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))
    states = torch.stack([s.step().state.clone() for _ in range(150)])

    with visualize(states, space, 'docs/gate.gif') as _:
        pass


@pytest.mark.parametrize('n', [30, 60])
def test_walkway(n):
    torch.manual_seed(42)

    pos_left = ((torch.rand((n, 2)) - 0.5) * 2.0) * torch.tensor([25.0, 5.0])
    pos_right = ((torch.rand((n, 2)) - 0.5) * 2.0) * torch.tensor([25.0, 5.0])

    ones = torch.ones((n, 1))
    zeros = torch.zeros((n, 1))

    x_vel_left = torch.normal(1.34 * ones, 0.26)
    x_vel_right = torch.normal(-1.34 * ones, 0.26)
    x_destination_left = 100.0 * ones
    x_destination_right = -100.0 * ones

    state_left = torch.cat(
        (pos_left, x_vel_left, zeros, x_destination_left, zeros), dim=-1)
    state_right = torch.cat(
        (pos_right, x_vel_right, zeros, x_destination_right, zeros), dim=-1)
    initial_state = torch.cat((state_left, state_right))

    space = [
        torch.tensor([(x, 5) for x in np.linspace(-25, 25, num=5000)]),
        torch.tensor([(x, -5) for x in np.linspace(-25, 25, num=5000)]),
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))
    states = []
    for _ in range(250):
        state = s.step().state
        # periodic boundary conditions
        state[state[:, 0] > 25, 0] -= 50
        state[state[:, 0] < -25, 0] += 50

        states.append(state.clone())
    states = torch.stack(states)

    with visualize(states, space, 'docs/walkway_{}.gif'.format(n)) as _:
        pass

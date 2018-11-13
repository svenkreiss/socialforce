import torch
import socialforce


def test_crossing():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
        [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],
    ])
    s = socialforce.Simulator(initial_state)
    states = torch.stack([s.step().state.clone() for _ in range(50)]).detach().numpy()

    # visualize
    print('')
    with socialforce.show.canvas('docs/crossing.png') as ax:
        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_narrow_crossing():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.5, 0.5, 2.0, 10.0],
        [2.0, 0.3, -0.5, 0.5, 0.0, 10.0],
    ])
    s = socialforce.Simulator(initial_state)
    states = torch.stack([s.step().state.clone() for _ in range(33)]).detach().numpy()

    # visualize
    print('')
    with socialforce.show.canvas('docs/narrow_crossing.png', figsize=(3, 5)) as ax:
        initial_state_np = initial_state.numpy()
        ax.plot(initial_state_np[:, 0], initial_state_np[:, 1],
                'x', color='grey', label='start')
        ax.plot(initial_state_np[:, -2], initial_state_np[:, -1],
                'o', color='grey', label='goal')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_opposing():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    s = socialforce.Simulator(initial_state, delta_t=0.2)
    states = torch.stack([s.step().state.clone() for _ in range(42)]).detach().numpy()

    # visualize
    print('')
    with socialforce.show.canvas('docs/opposing.png', figsize=(3, 5)) as ax:
        initial_state_np = initial_state.numpy()
        ax.plot(initial_state_np[:, 0], initial_state_np[:, 1],
                'x', color='grey', label='start')
        ax.plot(initial_state_np[:, -2], initial_state_np[:, -1],
                'o', color='grey', label='goal')

        for ped in range(2):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()


def test_2opposing():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 0.5, 0.0, 10.0],
        [0.6, 10.0, 0.0, -0.5, 0.6, 0.0],
        [2.0, 10.0, 0.0, -0.5, 2.0, 0.0],
    ])
    s = socialforce.Simulator(initial_state)
    states = torch.stack([s.step().state.clone() for _ in range(40)]).detach().numpy()

    # visualize
    print('')
    with socialforce.show.canvas('docs/2opposing.png') as ax:
        for ped in range(3):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.legend()

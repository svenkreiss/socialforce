"""Utility functions for plots and animations."""

from contextlib import contextmanager

import numpy as np
import torch

from . import potentials

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
except ImportError:
    plt = None
    mpl_animation = None


@contextmanager
def track_canvas(image_file=None, show=True, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    def format_ax(this_ax):
        if isinstance(this_ax, np.ndarray):
            for a in this_ax:
                format_ax(a)
            return

        this_ax.grid(linestyle='dotted')
        this_ax.set_aspect(1.0, 'datalim')
        this_ax.set_axisbelow(True)
        this_ax.set_xlabel('x [m]')
        this_ax.set_ylabel('y [m]')

    format_ax(ax)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def canvas(image_file=None, show=True, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def animation(n, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_axisbelow(True)

    context = {'fig': fig, 'ax': ax, 'update_function': None}
    yield context

    ani = mpl_animation.FuncAnimation(fig, context['update_function'], range(n))
    if movie_file:
        ani.save(movie_file, writer=writer)
    fig.show()
    plt.close(fig)


def states(ax, states, *, labels=None, **kwargs):  # pylint: disable=redefined-outer-name
    states = np.asarray(states)

    initial_state_np = states[0]
    ax.plot(initial_state_np[:, 0], initial_state_np[:, 1],
            'x', color='grey', label='start')
    ax.plot(initial_state_np[:, 6], initial_state_np[:, 7],
            'o', color='grey', label='goal')

    for ped in range(states.shape[1]):
        x = states[1:, ped, 0]
        y = states[1:, ped, 1]
        label = 'ped {}'.format(ped)
        if labels:
            label = labels[ped]
        ax.plot(x, y, '-o', label=label, markersize=2.5, **kwargs)


def potential1D(V, ax1, ax2=None, **kwargs):
    b = np.linspace(0, 3, 200)
    y = V.value_b(torch.from_numpy(b)).detach().numpy()
    y -= y[-1]

    ax1.set_xlabel('$b$ [m]')
    ax1.set_ylabel('$V$')
    ax1.plot(b, y, **kwargs)
    ax1.legend()

    if ax2 is not None:
        ax2.set_xlabel(r'$b$ [m]')
        ax2.set_ylabel(r'$\nabla V$')
        delta_b = b[1:] - b[:-1]
        average_b = 0.5 * (b[:-1] + b[1:])
        diff_b = y[1:] - y[:-1]
        ax2.plot(average_b, diff_b / delta_b, **kwargs)
        ax2.legend()


def potential1D_parametric(V, ax1, ax2=None, label=None, sigma_label=None, linestyle=None, **kwargs):
    potential1D(V, ax1, ax2, linestyle=linestyle, label=label, **kwargs)
    ax1.axvline(V.sigma, linestyle='dotted', label=sigma_label, **kwargs)
    if ax2 is not None:
        ax2.axvline(V.sigma, linestyle='dotted', label=sigma_label, **kwargs)


def potential2D(V, ax, nx=601, ny=601, **kwargs):
    # the "pedestrian of interest" is beta and the probe pedestrians are alpha

    x1 = np.linspace(-0.5, 1.5, nx, dtype=np.float32)
    x2 = np.linspace(-0.75, 0.75, ny, dtype=np.float32)
    xx1, xx2 = np.meshgrid(x1, x2)
    r_ab_probe = torch.from_numpy(np.stack((xx1, xx2), axis=-1)).view(-1, 2).unsqueeze(1)

    speeds_b = torch.ones((r_ab_probe.shape[1],)) * 1.0
    desired_directions_b = torch.zeros((r_ab_probe.shape[1], 2))
    desired_directions_b[:, 0] = 1.0

    ax.axhline(0.0, ls='dotted', color='gray')
    ax.axvline(0.0, ls='dotted', color='gray')

    with torch.no_grad():
        values = V.value_r_ab(r_ab_probe, speeds_b, desired_directions_b, 1.0)
        values -= torch.min(values)
    values = values.view((len(x1), len(x2)))

    ax.clabel(
        ax.contour(x1, x2, values, levels=np.linspace(0.2, 3.0, 15), vmax=1.0),
        inline=1, fontsize=10)

    ax.plot([0.0], [0.0], 'o', label='pedestrian', markersize=5.0, color='black')
    ax.set_xlabel('$x_1$ [m]')
    ax.set_ylabel('$x_2$ [m]')
    ax.set_aspect('equal')
    ax.legend()


def potential2Dgrad(V, ax, nx=601, ny=601, **kwargs):
    # the "pedestrian of interest" is beta and the probe pedestrians are alpha

    x1 = np.linspace(-0.5, 1.5, nx, dtype=np.float32)
    x2 = np.linspace(-0.75, 0.75, ny, dtype=np.float32)
    xx1, xx2 = np.meshgrid(x1, x2)
    r_ab_probe = torch.from_numpy(np.stack((xx1, xx2), axis=-1)).view(-1, 2).unsqueeze(1)

    speeds_b = torch.ones((r_ab_probe.shape[1],)) * 1.0
    desired_directions_b = torch.zeros((r_ab_probe.shape[1], 2))
    desired_directions_b[:, 0] = 1.0

    ax.axhline(0.0, ls='dotted', color='gray')
    ax.axvline(0.0, ls='dotted', color='gray')

    with torch.no_grad():
        grad = V.grad_r_ab_(r_ab_probe, speeds_b, desired_directions_b, 1.0)
        values = torch.linalg.norm(grad, ord=2, dim=-1)
        values -= torch.min(values)
    values = values.view((len(x1), len(x2)))

    ax.clabel(
        ax.contour(x1, x2, values, levels=np.linspace(0, 3.0, 16), vmax=1.5),
        inline=1, fontsize=10)

    ax.plot([0.0], [0.0], 'o', label='pedestrian', markersize=5.0, color='black')
    ax.set_xlabel('$x_1$ [m]')
    ax.set_ylabel('$x_2$ [m]')
    ax.set_aspect('equal')
    ax.legend()

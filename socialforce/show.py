"""Utility functions for plots and animations."""

from contextlib import contextmanager

import numpy as np

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

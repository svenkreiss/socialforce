import matplotlib.lines
import numpy as np

import socialforce


def test_loss_landscape_1pred():
    target = 1.0
    with socialforce.show.graph('docs/onedimensional_1pred_scenario.png', figsize=(6, 1.5)) as ax:
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.set_ylim(-0.1, 0.5)
        ax.set_xlim(-0.1, 2.1)
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, _ = ax.get_yaxis().get_view_interval()
        ax.add_artist(matplotlib.lines.Line2D(
            (xmin, xmax), (ymin, ymin), color='black', linewidth=2))

        ax.plot([0.0], [0.0], 'x', label='last ped position', markersize=5.0, color='grey')
        ax.plot([1.0], [0.0], 'o', label='true target position', markersize=5.0, color='grey')
        ax.plot([0.8], [0.0], 'o', label='predicted position', markersize=5.0, color='navy')
        ax.legend()

    x = np.linspace(0.0, 2.0, 101)
    loss_landscape = np.abs(x - target)
    with socialforce.show.graph('docs/onedimensional_1pred.png') as ax:
        ax.plot(x, loss_landscape, '-', label='1-step loss', color='navy')
        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('loss value')


def test_loss_landscape_2pred():
    print()
    target = 1.0
    with socialforce.show.graph('docs/onedimensional_2pred_scenario.png', figsize=(6, 2.0)) as ax:
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.set_ylim(-0.1, 0.5)
        ax.set_xlim(-0.1, 2.1)
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, _ = ax.get_yaxis().get_view_interval()
        ax.add_artist(matplotlib.lines.Line2D(
            (xmin, xmax), (ymin, ymin), color='black', linewidth=2))

        ax.plot([0.0], [0.0], 'x', label='last ped position', markersize=5.0, color='grey')
        ax.plot([0.5], [0.0], 'o', label='true target position 1', markersize=2.5, color='grey')
        ax.plot([0.8], [0.0], 'o', label='predicted position 1', markersize=2.5, color='orange')
        ax.plot([1.0], [0.0], 'o', label='true target position 2', markersize=5.0, color='grey')
        ax.plot([1.1], [0.0], 'o', label='predicted position 2', markersize=5.0, color='navy')
        ax.legend()

    x1 = np.linspace(0.0, 2.0, 101)
    x2 = np.linspace(-1.0, 1.0, 101)
    xx1, xx2 = np.meshgrid(x1, x2, sparse=True)
    loss = np.abs(target - (xx1 + xx2))
    with socialforce.show.graph('docs/onedimensional_2pred_fde.png') as ax:
        # ax.imshow(loss_landscape, '-', label='2-step loss', color='navy')
        c = ax.contour(x1, x2, loss, 10)
        ax.clabel(c, inline=1, fontsize=10)

        ax.plot([0.5], [0.5], 'o', label='true target position', markersize=5.0, color='grey')

        ax.set_xlabel('$x_1$ [m]')
        ax.set_ylabel('$x_2$ [m]')
        ax.legend()

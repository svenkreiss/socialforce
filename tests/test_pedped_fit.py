import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import scipy.optimize
import socialforce


def test_opposing():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)])

    # training
    def f(x):
        v0 = float(x[0])
        sigma_v = float(x[1])
        V = socialforce.PedPedPotential(0.4, v0, sigma_v)
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])

        # average euclidean distance loss
        diff = g - truth
        loss = torch.mul(diff, diff).mean()

        return loss

    parameters = np.array([1.2, 0.1])
    res = scipy.optimize.minimize(f, parameters, method='BFGS', options={'eps': 1e-3})
    print(res)


def test_opposing_mlp():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)])

    def potential(x):
        lin1_weight = torch.from_numpy(x[0:5]).float().unsqueeze(1)
        lin1_bias = torch.from_numpy(x[5:10]).float()
        lin2_weight = torch.from_numpy(x[10:15]).float().unsqueeze(0)
        lin2_bias = torch.from_numpy(x[15:16]).float()
        return socialforce.PedPedPotentialMLP(0.4, [
            (lin1_weight, lin1_bias),
            (lin2_weight, lin2_bias),
        ])

    # training
    def f(x):
        V = potential(x)
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])

        # average euclidean distance loss
        diff = g - truth
        loss = torch.mul(diff, diff).sum(dim=-1).mean()

        # constraint for V(3.0) = 0.0
        v_inf = V.value_b(torch.tensor([[3.0]]))[0]
        loss += 0.1 * torch.mul(v_inf, v_inf).item()

        return float(loss)

    np.random.seed(42)
    parameters = np.concatenate((
        np.random.rand(5) * 0.1,
        np.zeros(5),
        np.random.rand(5) * 0.1,
        np.zeros(1),
    ))
    initial_parameters = parameters.copy()
    res = scipy.optimize.minimize(f, parameters, method='L-BFGS-B', options={'eps': 1e-4, 'gtol': 1e-4})
    print(res)

    # make a plot of result
    V_initial = potential(initial_parameters)
    V = potential(res.x)
    b = np.linspace(0, 3)
    y_ref = 2.3 * np.exp(-1.0 * b / 0.3)
    y_initial = V_initial.value_b(torch.from_numpy(b).float()).detach().numpy()
    y_mlp = V.value_b(torch.from_numpy(b).float()).detach().numpy()

    fig, ax = plt.subplots()
    ax.set_xlabel('$b$ [m]')
    ax.set_ylabel('$V$')
    ax.plot(b, y_ref, label=r'$V_0 e^{-b/\sigma}$')
    ax.plot(b, y_initial, label=r'untrained MLP($b$)',
            linestyle='dashed', color='black')
    ax.plot(b, y_mlp, label=r'MLP($b$)')
    ax.axvline(0.3, color='black', linestyle='dotted')
    ax.legend()
    fig.savefig('docs/mlp_v.png', dpi=300)
    fig.show()

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$b$ [m]')
    ax.set_ylabel(r'$\nabla V$')
    delta_b = b[1:] - b[:-1]
    average_b = 0.5 * (b[:-1] + b[1:])
    ax.plot(average_b, (y_ref[1:] - y_ref[:-1]) / delta_b, label=r'$V_0 e^{-b/\sigma}$')
    ax.plot(average_b, (y_initial[1:] - y_initial[:-1]) / delta_b, label=r'untrained MLP($b$)',
            linestyle='dashed', color='black')
    ax.plot(average_b, (y_mlp[1:] - y_mlp[:-1]) / delta_b, label=r'MLP($b$)')
    ax.axvline(0.3, color='black', linestyle='dotted')
    ax.set_ylim(-4.9, 0.5)
    ax.legend()
    fig.savefig('docs/mlp_gradv.png', dpi=300)
    fig.show()

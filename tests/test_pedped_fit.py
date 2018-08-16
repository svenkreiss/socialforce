import matplotlib.pyplot as plt
import numpy as np
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
    torch.manual_seed(42)

    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)])

    V = socialforce.PedPedPotentialMLP(0.4)

    # training
    def f(x):
        V.set_parameters(torch.tensor(x))
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])

        # average euclidean distance loss
        diff = g - truth
        loss = torch.mul(diff, diff).sum(dim=-1).mean()

        # constraint for V(3.0) = 0.0
        v_inf = V.value_b(torch.tensor([[3.0]]))[0]
        loss += 0.1 * torch.mul(v_inf, v_inf).item()

        return float(loss)

    parameters = V.get_parameters().detach().numpy()
    initial_parameters = parameters.copy()
    res = scipy.optimize.minimize(f, parameters, method='L-BFGS-B',
                                  options={'eps': 1e-4, 'gtol': 1e-4})
    print(res)

    # make plots of result
    b = np.linspace(0, 3)
    y_ref = 2.3 * np.exp(-1.0 * b / 0.3)
    V.set_parameters(torch.tensor(initial_parameters))
    y_initial = V.value_b(torch.from_numpy(b).float()).detach().numpy()
    V.set_parameters(torch.tensor(res.x))
    y_mlp = V.value_b(torch.from_numpy(b).float()).detach().numpy()

    with socialforce.show.graph('docs/mlp_v.png') as ax:
        ax.set_xlabel('$b$ [m]')
        ax.set_ylabel('$V$')
        ax.plot(b, y_ref, label=r'$V_0 e^{-b/\sigma}$', color='C0')
        ax.axvline(0.3, color='black', linestyle='dotted', label=r'$\sigma$')
        ax.plot(b, y_initial, label=r'untrained MLP($b$)',
                linestyle='dashed', color='black')
        ax.plot(b, y_mlp, label=r'MLP($b$)', color='orange')
        ax.legend()

    with socialforce.show.graph('docs/mlp_gradv.png') as ax:
        ax.set_xlabel(r'$b$ [m]')
        ax.set_ylabel(r'$\nabla V$')
        delta_b = b[1:] - b[:-1]
        average_b = 0.5 * (b[:-1] + b[1:])
        ax.plot(average_b, (y_initial[1:] - y_initial[:-1]) / delta_b, label=r'untrained MLP($b$)',
                linestyle='dashed', color='black')
        ax.plot(average_b, (y_mlp[1:] - y_mlp[:-1]) / delta_b, label=r'MLP($b$)', color='orange')
        ax.plot(average_b, (y_ref[1:] - y_ref[:-1]) / delta_b, label=r'$V_0 e^{-b/\sigma}$', color='C0')
        ax.axvline(0.3, color='black', linestyle='dotted', label=r'$\sigma$')
        ax.set_ylim(-4.9, 0.5)
        ax.legend()

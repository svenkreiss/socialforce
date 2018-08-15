"""Test field-of-view computations."""

import torch
from socialforce.fieldofview import FieldOfView


def test_w():
    assert FieldOfView()(
        torch.tensor([
            [1.0, 0.0],
            [-1.0, 0.0],
        ]),
        torch.tensor([
            [1.0, 1.0],
            [-1.0, 1.0],
        ])
    ).tolist() == [1, 1]

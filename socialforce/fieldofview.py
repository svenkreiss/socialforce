"""Field of view computation."""

import math
import torch


class FieldOfView:
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """
    def __init__(self, twophi=200.0, out_of_view_factor=0.5):
        self.cosphi = math.cos(twophi / 2.0 / 180.0 * math.pi)
        self.out_of_view_factor = out_of_view_factor

    def __call__(self, e, f):
        """Weighting factor for field of view.

        e is rank 2 and normalized in the last index.
        f is a rank 3 tensor.
        """
        in_sight = torch.einsum('aj,abj->ab', (e, f)) > torch.norm(f, dim=-1) * self.cosphi
        out = torch.full_like(in_sight, self.out_of_view_factor, dtype=e.dtype)
        out[in_sight] = 1.0
        torch.diagonal(out)[:] = 0.0
        return out

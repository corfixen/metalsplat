"""Metal tile bounds for Hexels (metalsplat.kernels.project_hexels): the
per-primitive hexagon footprint `reference.hexels_ref.hexel_rects`
computes in ~30 small torch ops, in one kernel launch. Non-differentiable.
Validated against the reference in tests/test_rasterize_hexels.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel


@torch.no_grad()
def hexel_rects(
    transform: torch.Tensor,  # (N, 3, 3) or (N, 9)
    opacities: torch.Tensor,  # (N,)
    softness3: torch.Tensor,  # (N, 3)
    depths: torch.Tensor,  # (N,)
    img_width: int,
    img_height: int,
    near: float,
    conservative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 4) rects (EMPTY_RECT where culled) and (N,) bool valid, exactly
    as `reference.hexels_ref.project_hexels` computes them."""
    device = transform.device
    n = transform.shape[0]
    rects = torch.empty(n, 4, device=device, dtype=torch.float32)
    valid = torch.empty(n, device=device, dtype=torch.int32)
    if n > 0:
        fp = torch.tensor(
            [near, float(img_width), float(img_height), float(n), float(conservative)],
            device=device,
            dtype=torch.float32,
        )
        load_kernel("project_hexels").hexel_rects(
            transform.reshape(n, 9).float().contiguous(),
            opacities.float().contiguous(),
            softness3.float().contiguous(),
            depths.float().contiguous(),
            fp,
            rects,
            valid,
            threads=n,
        )
    return rects, valid.bool()

"""Pure-PyTorch reference for 3D Hexels (Honeycomb-Surfels): the executable
spec and gradient oracle for ``kernels/rasterize_hexels.metal``.

A Hexel (Kaushik, Paul and Yuille 2026, "Honeycomb-Surfels: Compact
Hexagonal Primitives for Efficient Scene Rendering and Reconstruction") is
a planar surfel like a 2DGS splat -- a center, a quaternion whose first two
rotation columns span the tangent plane, and two tangent scales -- but its
footprint is a regular pointy-top *hexagon* with a soft boundary instead of
a gaussian, and its color varies across the primitive:

    alpha(u, v) = min(0.99, opacity * sigmoid(-softness_e * sdf(u, v)))
    color(u, v) = c + u * g_u + v * g_v  [+ u^2 q_uu + u v q_uv + v^2 q_vv]

``(u, v)`` is the exact ray-plane intersection in the primitive's local,
scale-normalized frame, solved per pixel with the same homogeneous
"cross of two pulled-back planes" trick 2DGS uses (see
``project_2dgs_ref``). ``sdf`` is the hexagon's signed distance written as
the max of three oriented half-plane distances (one horizontal edge pair,
two slanted pairs); ``softness_e`` is the softness of the edge family that
wins that max, so an anisotropic Hexel can have three different edge
sharpnesses.

Every formula and constant here is transcribed from the CUDA reference
implementation (honeycomb-surfels/hexel-kernel,
``gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu``: ``preprocess_hexels3d``,
``compute_hex_bounds``, ``hex_sdf_edge``, ``bounded_quadratic_coordinates``
and the ``render_hexels3d`` template), with two convention changes that make
it fit this package:

- Cameras are this package's pinhole ``Camera`` (``x_pix = fx*x/z + cx``,
  pixel centers at ``i + 0.5``) rather than 3DGS view/projection matrices
  with an ``(W-1)/2`` NDC-to-pixel offset. The transform rows are therefore
  ``project_2dgs_ref``'s, not glm's transposed ``T``; the per-pixel math is
  otherwise identical (``k = -hu``, ``l = -hv`` there, and
  ``cross(k, l) == cross(hu, hv)``).
- View-dependent color terms (V2 linear, V3 degree-3 SH, and the specular
  "lobe") are per-primitive constants -- they depend on the direction to the
  primitive's center, never on the pixel -- so they are folded into the base
  color in plain PyTorch before rasterizing (``view_color_residual``). That
  is exactly what the hexels trainer already does for V2/V3; the lobe term
  is folded the same way here instead of inside the kernel.

Deliberately not reproduced (diagnostic or opt-in in hexel-kernel, and
disabled by every release training recipe): the edge prefilter
(``prefilter_edge_on``), exact-sort / pixel-risk kernels, child-primitive
packing and its ``overdraw_base_count`` split. Tile bounds are the exact
hexagon bounds (CUDA's ``conservative_bounds=True``), without the historical
intersection with the older ellipse AABB; the two only differ for a
primitive that crosses the camera plane.

This module brute-forces every primitive at every pixel (no tiling), so it
is only meant for small scenes and tests. It runs on any device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from metalsplat.reference.tiling_ref import EMPTY_RECT
from metalsplat.utils.quaternion import quat_to_rotmat

# Constants, verbatim from hexels3d_forward.cu.
NEAR_PLANE = 0.2
FAR_PLANE = 100.0  # only normalizes depth for the distortion regularizer
HEX_RADIUS = 2.0 / math.sqrt(3.0)  # circumradius of the unit-apothem hexagon
LOG_254 = math.log(254.0)
ALPHA_MAX = 0.99
ALPHA_MIN = 1.0 / 255.0
T_MIN = 1e-4
SOFTNESS_MIN = 1e-4
HEX_SLANT = 0.5 * math.sqrt(3.0)  # 0.8660254

COLOR_MODELS = ("flat", "linear", "quadratic", "quadratic_bounded")
VIEW_MODELS = ("none", "linear", "sh3", "lobe")

_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
SH3_COEFFS_BY_DEGREE = (0, 3, 8, 15)


def color_model_index(color_model: str) -> int:
    if color_model not in COLOR_MODELS:
        raise ValueError(
            f"color_model must be one of {COLOR_MODELS}, got {color_model!r}"
        )
    return COLOR_MODELS.index(color_model)


# --------------------------------------------------------------------------
# Hexagon SDF


def hex_sdf_edge(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance to the unit-apothem pointy-top hexagon and the id of
    the winning edge family: 0 = horizontal (|v|), 1 = first slant
    (|a u + b v|), 2 = second slant (|a u - b v|).

    Tie-breaking matches CUDA: the second slant wins a slant tie, and
    either slant wins a tie against the horizontal edge.
    """
    d0 = v.abs()
    d1 = (HEX_SLANT * u + 0.5 * v).abs()
    d2 = (HEX_SLANT * u - 0.5 * v).abs()
    second = d2 >= d1
    slant = torch.where(second, d2, d1)
    horizontal = d0 > slant
    dist = torch.where(horizontal, d0, slant) - 1.0
    edge = torch.where(
        horizontal,
        torch.zeros_like(u, dtype=torch.long),
        torch.where(second, 2, 1).to(torch.long),
    )
    return dist, edge


def bounded_quadratic_coordinates(
    u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``quadratic_bounded``'s coordinates: (u, v) radially pulled back onto
    the hexagon when outside it (``rho = sdf + 1 > 1``), so the quadratic
    color term cannot blow up in a primitive's soft fringe."""
    dist, _ = hex_sdf_edge(u, v)
    rho = dist + 1.0
    outside = rho > 1.0
    inv = torch.where(
        outside, 1.0 / torch.where(outside, rho, torch.ones_like(rho)), 1.0
    )
    return u * inv, v * inv


# --------------------------------------------------------------------------
# Projection (differentiable; also the production path, see rendering.py)


@dataclass
class HexelProjection:
    transform: (
        torch.Tensor
    )  # (N, 3, 3) rows: x-numerator, y-numerator, z; cols: t_u, t_v, mean
    depths: torch.Tensor  # (N,) camera-space z of the center (sort key)
    means2d: torch.Tensor  # (N, 2) projected center, pixels
    normal: torch.Tensor  # (N, 3) unit world normal flipped to face the camera
    raw_normal: torch.Tensor  # (N, 3) unit world normal, unflipped (rotation column 2)
    view_dirs: torch.Tensor  # (N, 3) unit direction camera -> center
    rects: torch.Tensor  # (N, 4) screen footprint xmin, ymin, xmax, ymax (no grad)
    valid: torch.Tensor  # (N,) bool (no grad)


def project_hexels(
    means: torch.Tensor,  # (N, 3)
    scales: torch.Tensor,  # (N, 2) positive
    quats: torch.Tensor,  # (N, 4) unit, (w, x, y, z)
    opacities: torch.Tensor,  # (N,) in [0, 1]
    softness3: torch.Tensor,  # (N, 3) per-edge softness, already floored
    R_wc: torch.Tensor,
    t_wc: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_width: int,
    img_height: int,
    near: float = NEAR_PLANE,
    scale_modifier: float = 1.0,
    rects_fn=None,
    conservative_bounds: bool = False,
) -> HexelProjection:
    """Builds the per-primitive data the rasterizer needs.

    ``conservative_bounds`` selects the tile bounds, as hexel-kernel's flag
    of the same name does (see `hexel_rects`): False (its default, and what
    every published checkpoint was trained with) clips each primitive to
    the historical ellipse window; True uses the exact hexagon bounds.

    ``rects_fn(transform, opacities, softness3, depths, w, h, near,
    conservative)``, if
    given, replaces the torch tile bounds below and returns ``(rects,
    valid)`` (``render_hexels`` passes the Metal kernel,
    ``ops.project_hexels.hexel_rects``).

    ``transform`` is ``project_2dgs_ref``'s ``M = W @ H`` (the same 9
    independent floats) with the tangent columns scaled by
    ``scale_modifier * scales``, so the rasterizer's ``(u, v)`` are in
    scale-normalized units where the hexagon has unit apothem.
    """
    mean_cam = means @ R_wc.T + t_wc
    depths = mean_cam[:, 2]
    z_safe = depths.clamp_min(near)
    means2d = torch.stack(
        [fx * mean_cam[:, 0] / z_safe + cx, fy * mean_cam[:, 1] / z_safe + cy], dim=-1
    )

    rotmat = quat_to_rotmat(quats)
    s = scales * scale_modifier
    tu_cam = (rotmat[..., :, 0] * s[:, 0:1]) @ R_wc.T
    tv_cam = (rotmat[..., :, 1] * s[:, 1:2]) @ R_wc.T
    raw_normal = rotmat[..., :, 2]

    cam_pos = -(R_wc.T @ t_wc)
    delta = means - cam_pos
    # CUDA: rsqrtf(|d|^2 + 1e-8).
    view_dirs = delta * torch.rsqrt((delta * delta).sum(-1, keepdim=True) + 1e-8)
    # surface_normal_sign: flip when the normal points away from the camera.
    flip = (raw_normal * view_dirs).sum(-1, keepdim=True) > 0
    normal = torch.where(flip, -raw_normal, raw_normal)

    row0 = torch.stack(
        [
            fx * tu_cam[:, 0] + cx * tu_cam[:, 2],
            fx * tv_cam[:, 0] + cx * tv_cam[:, 2],
            fx * mean_cam[:, 0] + cx * mean_cam[:, 2],
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            fy * tu_cam[:, 1] + cy * tu_cam[:, 2],
            fy * tv_cam[:, 1] + cy * tv_cam[:, 2],
            fy * mean_cam[:, 1] + cy * mean_cam[:, 2],
        ],
        dim=-1,
    )
    row2 = torch.stack([tu_cam[:, 2], tv_cam[:, 2], mean_cam[:, 2]], dim=-1)
    transform = torch.stack([row0, row1, row2], dim=-2)

    if rects_fn is not None:
        with torch.no_grad():
            rects, valid = rects_fn(
                transform.detach(), opacities.detach(), softness3.detach(),
                depths.detach(), img_width, img_height, near, conservative_bounds,
            )  # fmt: skip
    else:
        with torch.no_grad():
            rects, has_rect = hexel_rects(
                transform,
                opacities.detach(),
                softness3.detach(),
                img_width,
                img_height,
                near,
                conservative=conservative_bounds,
            )
            valid = (
                (depths.detach() > near) & has_rect & (opacities.detach() >= ALPHA_MIN)
            )
            empty = torch.tensor(EMPTY_RECT, device=means.device, dtype=rects.dtype)
            rects = torch.where(valid[:, None], rects, empty)

    return HexelProjection(
        transform=transform,
        depths=depths,
        means2d=means2d,
        normal=normal,
        raw_normal=raw_normal,
        view_dirs=view_dirs,
        rects=rects,
        valid=valid,
    )


def support_apothem(opacities: torch.Tensor, softness3: torch.Tensor) -> torch.Tensor:
    """(N,) local apothem of the region where ``alpha >= 1/255``.

    ``opacity * sigmoid(-s d) >= 1/255`` iff ``d <= log(255 opacity - 1) / s``;
    the margin is taken at the softness that makes it largest (the minimum
    softness when the log is positive, the maximum when it is negative), as
    ``preprocess_hexels3d`` does, including its tiny relative slack and the
    pad ``compute_hex_bounds`` adds before rejecting a degenerate support.
    """
    s_min = softness3.min(dim=-1).values
    s_max = softness3.max(dim=-1).values
    eps = 1.0 + 8.0 * torch.finfo(torch.float32).eps
    arg = 255.0 * eps * opacities - 1.0
    above = opacities > ALPHA_MIN
    log_cutoff = torch.log(
        torch.where(above, arg, torch.ones_like(arg)).clamp_min(1e-30)
    )
    margin = torch.where(
        above,
        log_cutoff / torch.where(log_cutoff < 0, s_max, s_min),
        LOG_254 / s_min,
    )
    apothem = 1.0 + margin
    return apothem + 4e-5 * apothem.abs().clamp_min(1.0)


def hexel_rects(
    transform: torch.Tensor,  # (N, 3, 3)
    opacities: torch.Tensor,  # (N,)
    softness3: torch.Tensor,  # (N, 3)
    img_width: int,
    img_height: int,
    near: float = NEAR_PLANE,
    conservative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 4) screen rectangle of each Hexel's ``alpha >= 1/255`` support,
    and an (N,) flag for "has a non-empty on-screen rectangle".

    Port of ``compute_hex_bounds``: project the six vertices of the support
    hexagon, clip its edges against the near plane (the visible part of a
    convex polygon is bounded by its visible vertices plus the points
    where its edges cross the plane), pad by one pixel each side.

    Unless ``conservative``, the rectangle is then intersected with
    `legacy_tile_window`, exactly as hexel-kernel does by default "so
    existing checkpoints render identically". That clipping is not
    cosmetic: it removes real contributions from large primitives whose
    historical ellipse window is smaller than their hexagon (typically big
    background sheets near the camera plane), and models trained with it
    rely on it -- rendering the published DTU scan24 V1 checkpoint with
    exact bounds loses 2 dB of held-out PSNR (9.8 dB on one view).
    """
    n = transform.shape[0]
    device, dtype = transform.device, transform.dtype
    a = support_apothem(opacities, softness3)  # (N,)
    ok = a > 0
    r = HEX_RADIUS * a
    unit = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [-0.5, 1.0], [-1.0, 0.0], [-0.5, -1.0], [0.5, -1.0]],
        device=device,
        dtype=dtype,
    )
    verts = torch.stack(
        [unit[:, 0] * r[:, None], unit[:, 1] * a[:, None]], dim=-1
    )  # (N,6,2)
    hom = torch.cat([verts, torch.ones(n, 6, 1, device=device, dtype=dtype)], dim=-1)
    proj = torch.einsum("nij,nkj->nki", transform, hom)  # (N, 6, 3): x-num, y-num, z
    z = proj[..., 2]
    in_front = z >= near
    z_safe = torch.where(in_front, z, torch.ones_like(z))
    px = proj[..., 0] / z_safe
    py = proj[..., 1] / z_safe

    nxt = torch.roll(torch.arange(6, device=device), -1)
    za, zb = z, z[:, nxt]
    crosses = (za < near) != (zb < near)
    t = (near - za) / torch.where(crosses, zb - za, torch.ones_like(za))
    edge = hom + t[..., None] * (hom[:, nxt] - hom)
    edge_proj = torch.einsum("nij,nkj->nki", transform, edge)
    ex = edge_proj[..., 0] / near
    ey = edge_proj[..., 1] / near

    big = torch.tensor(float("inf"), device=device, dtype=dtype)
    xs = torch.cat(
        [torch.where(in_front, px, big), torch.where(crosses, ex, big)], dim=1
    )
    ys = torch.cat(
        [torch.where(in_front, py, big), torch.where(crosses, ey, big)], dim=1
    )
    lo_x, lo_y = xs.min(dim=1).values, ys.min(dim=1).values
    xs = torch.cat(
        [torch.where(in_front, px, -big), torch.where(crosses, ex, -big)], dim=1
    )
    ys = torch.cat(
        [torch.where(in_front, py, -big), torch.where(crosses, ey, -big)], dim=1
    )
    hi_x, hi_y = xs.max(dim=1).values, ys.max(dim=1).values

    any_point = torch.isfinite(lo_x) & torch.isfinite(hi_x)
    on_screen = (lo_x <= img_width) & (lo_y <= img_height) & (hi_x >= 0) & (hi_y >= 0)
    has = ok & any_point & on_screen
    rects = torch.stack([lo_x - 1.0, lo_y - 1.0, hi_x + 1.0, hi_y + 1.0], dim=-1)
    if not conservative:
        win_lo, win_hi, win_ok = legacy_tile_window(
            transform, softness3, opacities, img_width, img_height
        )
        rects = torch.cat(
            [torch.maximum(rects[:, :2], win_lo), torch.minimum(rects[:, 2:], win_hi)],
            -1,
        )
        has = has & win_ok & (rects[:, 2] >= rects[:, 0]) & (rects[:, 3] >= rects[:, 1])
    rects = torch.where(has[:, None], rects, torch.zeros_like(rects))
    return rects, has


LEGACY_TILE = 16  # hexel-kernel's fixed kBlockX/kBlockY
FILTER_SIZE = 0.707106  # kFilterSize


def legacy_tile_window(
    transform: torch.Tensor,  # (N, 3, 3), this package's pixel convention
    softness3: torch.Tensor,
    opacities: torch.Tensor,
    img_width: int,
    img_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hexel-kernel's historical tile window (``compute_aabb`` +
    ``get_rect`` in ``preprocess_hexels3d``): the bounding box of the
    perspective image of the circle of radius
    ``HEX_RADIUS * (1 + log(254) / min softness)``, joined with the
    screen-filter radius, in 16 px tiles.

    Returns continuous ``(lo, hi)`` (N, 2) bounds that select exactly those
    tiles under `bin_and_sort_rects` (a 16 px tile range ``[a, b)`` maps to
    ``[16 a + 0.5, 16 b - 0.5]``), and whether the window is non-empty
    (CUDA culls the primitive otherwise). Computed in CUDA's pixel frame,
    whose integer coordinates are pixel centers: this package's pixel
    ``x`` is CUDA's ``x - 0.5``, so the rows shift by ``-0.5 * row2``.
    """
    b = LEGACY_TILE
    r2 = transform[:, 2]
    r0 = transform[:, 0] - 0.5 * r2
    r1 = transform[:, 1] - 0.5 * r2
    cut = HEX_RADIUS * (1.0 + LOG_254 / softness3.min(dim=-1).values)
    t = torch.stack([cut * cut, cut * cut, -torch.ones_like(cut)], -1)
    d = (t * r2 * r2).sum(-1)
    bounded = d < 0
    nonzero = d != 0
    f = t / torch.where(nonzero, d, torch.ones_like(d))[:, None]
    p = torch.stack([(f * r0 * r2).sum(-1), (f * r1 * r2).sum(-1)], -1)
    h0 = p * p - torch.stack([(f * r0 * r0).sum(-1), (f * r1 * r1).sum(-1)], -1)
    h = h0.clamp_min(1e-4).sqrt()
    radius = torch.ceil(torch.maximum(h.max(dim=-1).values, cut * FILTER_SIZE))
    grid = torch.tensor(
        [(img_width + b - 1) // b, (img_height + b - 1) // b],
        device=transform.device,
        dtype=transform.dtype,
    )

    def get_rect(r):  # (N, 2) int radii; int casts truncate toward zero
        lo = torch.minimum(((p - r) / b).trunc().clamp_min(0), grid)
        hi = torch.minimum(((p + r + b - 1) / b).trunc().clamp_min(0), grid)
        return lo, hi

    rr = radius[:, None].expand(-1, 2)
    lo, hi = get_rect(rr)
    visible = nonzero & (lo < hi).all(-1)
    tight = torch.minimum(
        rr, torch.ceil(torch.maximum(h, (cut * FILTER_SIZE)[:, None])) + 1
    )
    use_tight = (bounded & (opacities <= 1.0))[:, None]
    lo, hi = get_rect(torch.where(use_tight, tight, rr))
    return lo * b + 0.5, hi * b - 0.5, visible


# --------------------------------------------------------------------------
# View-dependent color (folded into the base color before rasterizing)


def eval_sh3_residual(
    coeffs: torch.Tensor, dirs: torch.Tensor, degree: int = 3
) -> torch.Tensor:
    """(N, 3) V3 color residual: SH bands 1..degree, no DC term, no offset,
    no clamp -- hexels' ``_eval_sh3_residual``. ``coeffs`` is (N, 15, 3)."""
    if not 0 <= degree <= 3:
        raise ValueError(f"SH degree must be in [0, 3], got {degree}")
    x, y, z = dirs.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    basis = torch.stack(
        [
            -_SH_C1 * y,
            _SH_C1 * z,
            -_SH_C1 * x,
            _SH_C2[0] * xy,
            _SH_C2[1] * yz,
            _SH_C2[2] * (2.0 * zz - xx - yy),
            _SH_C2[3] * xz,
            _SH_C2[4] * (xx - yy),
            _SH_C3[0] * y * (3.0 * xx - yy),
            _SH_C3[1] * xy * z,
            _SH_C3[2] * y * (4.0 * zz - xx - yy),
            _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _SH_C3[4] * x * (4.0 * zz - xx - yy),
            _SH_C3[5] * z * (xx - yy),
            _SH_C3[6] * x * (xx - 3.0 * yy),
        ],
        dim=-1,
    )
    k = SH3_COEFFS_BY_DEGREE[degree]
    return (basis[:, :k].unsqueeze(-1) * coeffs[:, :k]).sum(dim=1)


def view_color_residual(
    view_model: str,
    view_dirs: torch.Tensor,  # (N, 3) unit, camera -> center
    raw_normal: torch.Tensor,  # (N, 3) unflipped unit normal
    view_grads: torch.Tensor | None,  # (N, 3, 3) linear or (N, 15, 3) SH
    specular_ks: torch.Tensor | None,  # (N, 3)
    specular_log_s: torch.Tensor | None,  # (N,)
    sh_degree: int = 3,
) -> torch.Tensor | None:
    """The per-primitive color offset of each view model, or None.

    - ``linear`` (V2): channel ``i`` gets ``view_grads[:, i, :] . dir``
      (hexel-kernel's native ``view_model="linear"``, equivalently the
      trainer's ``bmm`` fast path).
    - ``sh3`` (V3): ``eval_sh3_residual``.
    - ``lobe``: ``ks * max(-n . dir, 0) ** exp(clamp(log_s, 0, 5))`` with the
      *unflipped* normal, as in ``render_hexels3d``.
    """
    if view_model == "none":
        return None
    if view_model == "linear":
        return torch.einsum("nij,nj->ni", view_grads, view_dirs)
    if view_model == "sh3":
        return eval_sh3_residual(view_grads, view_dirs, sh_degree)
    if view_model == "lobe":
        cos_nv = (-(raw_normal * view_dirs).sum(-1)).clamp_min(0.0)
        s_exp = specular_log_s.clamp(0.0, 5.0).exp()
        positive = cos_nv > 0
        spec = torch.where(
            positive,
            torch.where(positive, cos_nv, torch.ones_like(cos_nv)) ** s_exp,
            0.0,
        )
        return specular_ks * spec[:, None]
    raise ValueError(f"view_model must be one of {VIEW_MODELS}, got {view_model!r}")


# --------------------------------------------------------------------------
# Rasterization (brute force)


@dataclass
class HexelRasterOutput:
    image: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W) 1 - final transmittance
    depth: torch.Tensor  # (H, W) alpha-weighted sum of intersection depth
    normal: torch.Tensor  # (H, W, 3) alpha-weighted sum of camera-facing normals
    distortion: torch.Tensor  # (H, W)
    median_depth: torch.Tensor  # (H, W)
    overdraw: torch.Tensor  # (H, W) sum of unclamped alpha over visited primitives
    overdraw_hits: torch.Tensor  # (H, W) count of visited primitives (no grad)
    contrib: torch.Tensor  # (N,) sum of compositing weights (no grad)
    owner_id: torch.Tensor  # (H, W) int64, -1 where nothing composited
    owner_weight: torch.Tensor  # (H, W)
    owner_uv: torch.Tensor  # (H, W, 2)
    owner_sdf: torch.Tensor  # (H, W)
    owner_depth: torch.Tensor  # (H, W)


def evaluate_color(
    color_model: int,
    colors: torch.Tensor,  # (..., 3)
    color_grads: torch.Tensor,  # (..., 6): g_u (3), g_v (3)
    color_quads: torch.Tensor,  # (..., 9): q_uu, q_uv, q_vv
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    c = colors
    if color_model >= 1:
        c = (
            c
            + u[..., None] * color_grads[..., 0:3]
            + v[..., None] * color_grads[..., 3:6]
        )
    if color_model >= 2:
        uq, vq = (u, v) if color_model == 2 else bounded_quadratic_coordinates(u, v)
        c = (
            c
            + (uq * uq)[..., None] * color_quads[..., 0:3]
            + (uq * vq)[..., None] * color_quads[..., 3:6]
            + (vq * vq)[..., None] * color_quads[..., 6:9]
        )
    return c


def rasterize_hexels(
    transform: torch.Tensor,  # (N, 3, 3)
    opacities: torch.Tensor,  # (N,)
    softness3: torch.Tensor,  # (N, 3) floored at SOFTNESS_MIN
    colors: torch.Tensor,  # (N, 3) base color incl. any view residual
    color_grads: torch.Tensor,  # (N, 6)
    color_quads: torch.Tensor,  # (N, 9)
    normals: torch.Tensor,  # (N, 3) camera-facing
    depths: torch.Tensor,  # (N,) sort key
    valid: torch.Tensor,  # (N,) bool
    img_width: int,
    img_height: int,
    color_model: int = 1,
    background: torch.Tensor | None = None,
    near: float = NEAR_PLANE,
) -> HexelRasterOutput:
    """Front-to-back compositing of every valid primitive at every pixel,
    sorted by center depth. Differentiable w.r.t. every float input."""
    device, dtype = transform.device, transform.dtype
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)
    h, w = img_height, img_width
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + 0.5,
        torch.arange(w, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    px, py = xs.reshape(-1), ys.reshape(-1)
    p = px.numel()
    n = transform.shape[0]

    zero = torch.zeros(p, device=device, dtype=dtype)
    color = torch.zeros(p, 3, device=device, dtype=dtype)
    depth = zero.clone()
    normal = torch.zeros(p, 3, device=device, dtype=dtype)
    T = torch.ones(p, device=device, dtype=dtype)
    a_sum = zero.clone()
    m1 = zero.clone()
    m2 = zero.clone()
    distortion = zero.clone()
    median = zero.clone()
    overdraw = zero.clone()
    overdraw_hits = zero.clone()
    done = torch.zeros(p, device=device, dtype=torch.bool)
    contrib = []
    owner_id = torch.full((p,), -1, device=device, dtype=torch.long)
    owner_w = zero.clone()
    owner_uv = torch.zeros(p, 2, device=device, dtype=dtype)
    owner_sdf = zero.clone()
    owner_depth = zero.clone()

    order = torch.argsort(depths.detach(), stable=True)
    for gid in order.tolist():
        if not bool(valid[gid]):
            contrib.append((gid, None))
            continue
        row0, row1, row2 = transform[gid, 0], transform[gid, 1], transform[gid, 2]
        hu = row0[None, :] - px[:, None] * row2[None, :]
        hv = row1[None, :] - py[:, None] * row2[None, :]
        cr = torch.linalg.cross(hu, hv)
        nondeg = cr[:, 2] != 0
        cz = torch.where(nondeg, cr[:, 2], torch.ones_like(cr[:, 2]))
        u = cr[:, 0] / cz
        v = cr[:, 1] / cz
        z = row2[0] * u + row2[1] * v + row2[2]
        dist, edge = hex_sdf_edge(u, v)
        soft = softness3[gid][edge]
        boundary = torch.sigmoid(-dist * soft)
        raw = opacities[gid] * boundary
        alpha = raw.clamp_max(ALPHA_MAX)
        hit = nondeg & (z >= near) & (alpha >= ALPHA_MIN) & ~done

        test_T = T * (1.0 - alpha)
        terminate = hit & (test_T < T_MIN)
        acc = hit & ~terminate
        wgt = torch.where(acc, alpha * T, zero)

        overdraw = overdraw + torch.where(hit, raw, zero)
        overdraw_hits = overdraw_hits + hit.to(dtype)

        c = evaluate_color(
            color_model, colors[gid], color_grads[gid], color_quads[gid], u, v
        )
        color = color + wgt[:, None] * c
        z_acc = torch.where(acc, z, torch.ones_like(z))
        median = torch.where(acc & (T > 0.5), z_acc, median)
        m = FAR_PLANE / (FAR_PLANE - near) * (1.0 - near / z_acc)
        distortion = distortion + wgt * (m * m * a_sum + m2 - 2.0 * m * m1)
        a_sum = a_sum + wgt
        m1 = m1 + wgt * m
        m2 = m2 + wgt * m * m
        depth = depth + wgt * z_acc
        normal = normal + wgt[:, None] * normals[gid][None, :]

        with torch.no_grad():
            better = acc & (wgt > owner_w)
            owner_id = torch.where(better, gid, owner_id)
            owner_w = torch.where(better, wgt, owner_w)
            owner_uv = torch.where(better[:, None], torch.stack([u, v], -1), owner_uv)
            owner_sdf = torch.where(better, dist, owner_sdf)
            owner_depth = torch.where(better, z, owner_depth)
        contrib.append((gid, wgt.detach().sum()))

        T = torch.where(acc, test_T, T)
        done = done | terminate

    contrib_t = torch.zeros(n, device=device, dtype=dtype)
    for gid, val in contrib:
        if val is not None:
            contrib_t[gid] = val

    image = color + T[:, None] * background[None, :]
    return HexelRasterOutput(
        image=image.reshape(h, w, 3),
        alpha=(1.0 - T).reshape(h, w),
        depth=depth.reshape(h, w),
        normal=normal.reshape(h, w, 3),
        distortion=distortion.reshape(h, w),
        median_depth=median.reshape(h, w),
        overdraw=overdraw.reshape(h, w),
        overdraw_hits=overdraw_hits.reshape(h, w),
        contrib=contrib_t,
        owner_id=owner_id.reshape(h, w),
        owner_weight=owner_w.reshape(h, w),
        owner_uv=owner_uv.reshape(h, w, 2),
        owner_sdf=owner_sdf.reshape(h, w),
        owner_depth=owner_depth.reshape(h, w),
    )

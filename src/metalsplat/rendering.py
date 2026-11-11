"""High-level glue: GaussianModel + Camera -> project -> rasterize -> image."""

from __future__ import annotations

from typing import NamedTuple

import torch

from metalsplat.camera import Camera
from metalsplat.filter3d import apply_3d_filter
from metalsplat.gaussians import GaussianModel
from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.hexels_model import HexelModel
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.project_2dgs import project_gaussians_2dgs
from metalsplat.ops.project_hexels import hexel_rects as metal_hexel_rects
from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.ops.rasterize_2dgs import rasterize_gaussians_2dgs
from metalsplat.ops.rasterize_hexels import (
    HexelOwner,
    rasterize_hexels,
    rasterize_hexels_owner,
)
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE
from metalsplat.reference.hexels_ref import (
    NEAR_PLANE,
    HexelProjection,
    project_hexels,
    view_color_residual,
)
from metalsplat.reference.rasterize_2dgs_ref import DEFAULT_FILTER_SIZE


class RenderAux(NamedTuple):
    """Auxiliary render outputs (see `render(..., return_aux=True)`)."""

    image: torch.Tensor  # (H, W, 3)
    means2d: torch.Tensor  # (N, 2), with retain_grad() called
    valid: torch.Tensor  # (N,)
    final_T: torch.Tensor  # (H, W) per-pixel transmittance, requires_grad=False
    depth: torch.Tensor  # (H, W) alpha-weighted expected depth, requires_grad=False
    radii: torch.Tensor  # (N,) projected 3-sigma pixel radius, 0 if culled


def render(
    model: GaussianModel,
    camera: Camera,
    near: float = 0.2,
    eps2d: float = 0.3,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: torch.Tensor | None = None,
    return_aux: bool = False,
    abs_grad_accum: torch.Tensor | None = None,
    near_fade: tuple[float, float] | None = None,
    antialias: bool = True,
    filter_3d: torch.Tensor | None = None,
):
    """Renders `model` from `camera`'s viewpoint. Returns an (H, W, 3) image.

    If `return_aux` is True, instead returns a `RenderAux` (a NamedTuple, so
    both `aux.depth` and positional unpacking work), with
    `means2d.retain_grad()` already called. `valid` is used by training
    loops to count how many steps each gaussian was visible for; `final_T`
    (per-pixel transmittance, close to 1 where ~no gaussian contributed) is
    used by metalsplat.seed to find uncovered regions; `depth` is the
    alpha-weighted expected depth (forward-only, no gradient); `radii` is
    each gaussian's projected pixel radius (0 if culled).

    `abs_grad_accum`, if given, is passed through to rasterize_gaussians:
    an (N,) tensor that backward() atomically adds each gaussian's
    |screen-space gradient contribution| into (AbsGS-style densification
    signal, metalsplat.densify) -- prefer this over means2d.grad.norm()
    for that purpose, since contributions from different pixels can have
    opposite signs and cancel out in means2d.grad's signed sum.

    `near_fade`, if given, is `(r0, r1)` in world units: a sphere around the
    camera inside which gaussians are suppressed. Opacity is scaled by a
    smoothstep that is 0 closer than `r0` and 1 beyond `r1`, so gaussians
    sitting in the empty space just in front of the camera stop contributing.
    Reconstruction puts spurious gaussians there -- they are only ever seen
    by a couple of training views, so nothing constrains them -- and because
    they are near the camera they sweep across the whole frame during a
    camera move, which is what makes an orbit flicker.

    The fade band is the point: a hard cut makes each gaussian vanish in a
    single frame as the camera crosses it, which trades flicker for popping.
    `r1` must be comfortably below the distance to the nearest real geometry
    (for an orbit, the camera's height above the ground) or this punches a
    hole in the scene instead.

    `antialias` (default True) scales each gaussian's opacity by the
    Mip-Splatting compensation factor, correcting for the energy the eps2d
    low-pass filter would otherwise add to sub-pixel gaussians. Set False to
    reproduce renders from before this existed, or to match a model trained
    without it.

    `filter_3d`, if given, is the (N,) per-gaussian filter radius from
    metalsplat.filter3d.compute_3d_filter: the *world-space* half of
    Mip-Splatting, band-limiting each gaussian to the finest detail any
    training camera resolved it at. Recompute it when the gaussian count
    changes, and keep it consistent between training and rendering -- a
    model trained with it expects to be rendered with it.
    """
    scales = model.scales
    opacities = model.opacities
    if filter_3d is not None:
        scales, opacities = apply_3d_filter(scales, opacities, filter_3d)

    means2d, depths, conics, radii, valid, compensation = project_gaussians(
        model.means,
        scales,
        model.quats,
        camera.R_wc,
        camera.t_wc,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.img_width,
        camera.img_height,
        near=near,
        eps2d=eps2d,
    )
    if return_aux and means2d.requires_grad:
        # No-op under torch.no_grad() / inference, where aux is still useful
        # for depth and coverage but there's no graph to retain a grad on.
        means2d.retain_grad()

    if antialias:
        # Mip-Splatting's anti-aliasing factor. project_gaussians dilates the
        # 2D covariance by eps2d as a low-pass filter; that also inflates the
        # integral of the gaussian's density, so a sub-pixel gaussian renders
        # stronger than it should and flickers as it crosses pixel
        # boundaries. `compensation` is the factor that takes that extra
        # energy back out of the peak opacity. ~1 for anything comfortably
        # larger than a pixel, falling to 0 for the smallest gaussians.
        opacities = opacities * compensation
    if near_fade is not None:
        r0, r1 = near_fade
        dist = (model.means - camera.position).norm(dim=-1)
        w = ((dist - r0) / max(r1 - r0, 1e-6)).clamp(0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep: flat at both ends, no crease
        opacities = opacities * w
        # Fully-faded gaussians contribute nothing, so drop them from tile
        # binning too rather than paying to composite a zero.
        valid = valid * (w > 0.0).to(valid.dtype)

    if model.sh_degree == 0:
        colors = model.colors
    else:
        view_dirs = torch.nn.functional.normalize(model.means - camera.position, dim=-1)
        colors = model.colors_from_view(view_dirs)

    result = rasterize_gaussians(
        means2d,
        depths,
        conics,
        opacities,
        colors,
        radii,
        valid,
        camera.img_width,
        camera.img_height,
        tile_size=tile_size,
        background=background,
        return_aux=return_aux,
        abs_grad_accum=abs_grad_accum,
    )
    if return_aux:
        image, depth, final_T = result
        return RenderAux(
            image=image,
            means2d=means2d,
            valid=valid,
            final_T=final_T,
            depth=depth,
            radii=radii,
        )
    return result


class Render2DGSAux(NamedTuple):
    """Auxiliary render outputs for `render_2dgs(..., return_aux=True)`.

    Unlike `RenderAux.depth` (3DGS, forward-only/no gradient), `depth` and
    `normal` here are genuinely differentiable -- 2DGS's normal-consistency
    and distortion losses need real gradients through them.
    """

    image: torch.Tensor  # (H, W, 3)
    means2d: torch.Tensor  # (N, 2), with retain_grad() called
    valid: torch.Tensor  # (N,)
    final_T: torch.Tensor  # (H, W) per-pixel transmittance, requires_grad=False
    depth: (
        torch.Tensor
    )  # (H, W) alpha-weighted ray-splat-intersection depth, differentiable
    normal: torch.Tensor  # (H, W, 3) alpha-weighted world-space normal, differentiable
    distortion: (
        torch.Tensor
    )  # (H, W) per-pixel Mip-NeRF-360/2DGS regularizer map, differentiable


def render_2dgs(
    model: Gaussian2DModel,
    camera: Camera,
    near: float = 0.2,
    filter_size: float = DEFAULT_FILTER_SIZE,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: torch.Tensor | None = None,
    return_aux: bool = False,
    abs_grad_accum: torch.Tensor | None = None,
    pixel_count_accum: torch.Tensor | None = None,
):
    """Renders `model` (a `Gaussian2DModel`) from `camera`'s viewpoint.

    A separate function from `render()` rather than a dispatch inside it:
    `render()`'s signature carries 3DGS-specific concepts that don't apply
    here (`filter_3d`/Mip-Splatting's `antialias` compensation -- 2DGS's
    own screen-space low-pass fallback, in `rasterize_2dgs.metal`, is a
    different mechanism), and this codebase has one concrete function per
    concern (one per kernel) rather than isinstance dispatch anywhere.

    Returns an (H, W, 3) image, or (if `return_aux`) a `Render2DGSAux` --
    see that NamedTuple's docstring for which fields carry real gradients.
    `distortion.mean()`-style reductions feed `metalsplat.losses.
    distortion_loss`; `depth`/`normal` feed
    `metalsplat.losses.normal_consistency_loss`, which also needs the
    accumulated opacity `1 - final_T` -- `depth` and `normal` are both
    alpha-weighted *sums*, so neither can be interpreted without it (see
    that function's docstring).

    `filter_size` is the pixel standard deviation of the rasterizer's own
    screen-space fallback for near-edge-on splats (see
    `metalsplat.reference.rasterize_2dgs_ref.DEFAULT_FILTER_SIZE`). It
    also widens every splat's screen footprint, so the projection's tile
    bounds use it too. (There is no `eps2d`: that dilated the 3DGS EWA
    covariance this path used to borrow for its tile bounds, which are now
    exact -- see `metalsplat.reference.project_2dgs_ref.surfel_rects`.)

    `abs_grad_accum`, if given, is passed through to
    `rasterize_gaussians_2dgs`: an (N,) tensor that backward() atomically
    adds each gaussian's AbsGS-style densification signal into -- see
    that function's docstring and `metalsplat.densify2dgs` for how it's
    used. Pass `pixel_count_accum` alongside it to also collect the
    covered-pixel counts that normalize that signal per pixel rather
    than per screen area.
    """
    opacities = model.opacities
    means2d, depths, rects, valid, transform, normal = project_gaussians_2dgs(
        model.means,
        model.scales,
        model.quats,
        camera.R_wc,
        camera.t_wc,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.img_width,
        camera.img_height,
        near=near,
        filter_size=filter_size,
        # Opacity-aware tile bounds, as 3DGS's binning does: a faint splat's
        # alpha clears the cutoff over a much smaller area.
        opacities=opacities.detach(),
    )
    if return_aux and means2d.requires_grad:
        means2d.retain_grad()

    if model.sh_degree == 0:
        colors = model.colors
    else:
        view_dirs = torch.nn.functional.normalize(model.means - camera.position, dim=-1)
        colors = model.colors_from_view(view_dirs)

    image, depth, out_normal, distortion, final_T = rasterize_gaussians_2dgs(
        means2d,
        transform,
        normal,
        opacities,
        colors,
        depths,
        rects,
        valid,
        camera.img_width,
        camera.img_height,
        tile_size=tile_size,
        near=near,
        filter_size=filter_size,
        background=background,
        abs_grad_accum=abs_grad_accum,
        pixel_count_accum=pixel_count_accum,
    )
    if return_aux:
        return Render2DGSAux(
            image=image,
            means2d=means2d,
            valid=valid,
            final_T=final_T,
            depth=depth,
            normal=out_normal,
            distortion=distortion,
        )
    return image


class RenderHexelsAux(NamedTuple):
    """Outputs of `render_hexels(..., return_aux=True)`.

    `image`, `alpha`, `depth`, `normal`, `distortion`, `median_depth` and
    `overdraw` are differentiable. `depth`/`normal` are alpha-weighted sums
    (as in 2DGS: divide by `alpha` for expected values), so they feed
    `metalsplat.losses.normal_consistency_loss(normal, depth, alpha,
    camera)` and `distortion_loss(distortion)` directly. `median_depth` is
    the depth of the last primitive composited while transmittance was
    above 0.5 (hexels' median-depth surface extraction). `overdraw` is the
    sum of unclamped alpha over every primitive a pixel visited (hexels'
    screen-repulsion signal); `overdraw_hits` counts them.
    """

    image: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W)
    depth: torch.Tensor  # (H, W)
    normal: torch.Tensor  # (H, W, 3)
    distortion: torch.Tensor  # (H, W)
    median_depth: torch.Tensor  # (H, W)
    overdraw: torch.Tensor  # (H, W)
    overdraw_hits: torch.Tensor  # (H, W), no grad
    final_T: torch.Tensor  # (H, W), no grad
    contrib: (
        torch.Tensor
    )  # (N,) summed compositing weight, no grad (zeros unless requested)
    means2d: torch.Tensor  # (N, 2) projected centers, pixels
    valid: torch.Tensor  # (N,) bool, binned for rasterization
    projection: HexelProjection


def _hexel_inputs(
    model: HexelModel,
    camera: Camera,
    near: float,
    scale_modifier: float,
    conservative_bounds: bool = False,
):
    opacities = model.opacities
    softness3 = model.softness3
    proj = project_hexels(
        model.means,
        model.scales,
        model.quats,
        opacities,
        softness3,
        camera.R_wc,
        camera.t_wc,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.img_width,
        camera.img_height,
        near=near,
        scale_modifier=scale_modifier,
        rects_fn=metal_hexel_rects if model.means.device.type == "mps" else None,
        conservative_bounds=conservative_bounds,
    )
    colors = model.colors
    residual = view_color_residual(
        model.view_model,
        proj.view_dirs,
        proj.raw_normal,
        getattr(model, "view_grads", None),
        getattr(model, "specular_ks", None),
        getattr(model, "specular_log_s", None),
        sh_degree=model.active_sh_degree,
    )
    if residual is not None:
        colors = colors + residual
    n = model.num_points
    args = (
        proj.transform.reshape(n, 9),
        opacities,
        softness3.contiguous(),
        colors,
        model.color_grads_flat(),
        model.color_quads_flat(),
        proj.normal,
        proj.depths,
        proj.rects,
        proj.valid,
        camera.img_width,
        camera.img_height,
    )
    return proj, args


def render_hexels(
    model: HexelModel,
    camera: Camera,
    background: torch.Tensor | None = None,
    near: float = NEAR_PLANE,
    scale_modifier: float = 1.0,
    tile_size: int = DEFAULT_TILE_SIZE,
    return_aux: bool = False,
    return_contrib: bool = False,
    abs_grad_accum: torch.Tensor | None = None,
    pixel_count_accum: torch.Tensor | None = None,
    xy_grad_accum: torch.Tensor | None = None,
    conservative_bounds: bool = False,
):
    """Renders a `HexelModel` from `camera` on MPS.

    `conservative_bounds` mirrors hexel-kernel's flag: False (default, as
    there) clips each primitive's tiles to the historical ellipse window,
    which models trained by hexels -- including the published checkpoints
    -- depend on; True uses the exact hexagon bounds (see
    `reference.hexels_ref.hexel_rects`).

    Returns the (H, W, 3) image, or a `RenderHexelsAux` with every
    rasterizer output if `return_aux`. View-dependent color (V2 linear, V3
    SH up to `model.active_sh_degree`, or the specular lobe) is evaluated
    per primitive here and folded into the base color, as hexels does, so
    gradients reach the primitive centers through the view direction.

    The projection is plain differentiable PyTorch (an O(N) handful of
    elementwise ops); only the per-pixel work runs in the Metal kernels.
    `return_contrib` fills `contrib` (for MCMC relocation). The
    accumulators are updated in place by backward(): `abs_grad_accum` and
    `pixel_count_accum` as in `render_2dgs`, `xy_grad_accum` (N, 2) with
    the signed screen-space gradient.
    """
    proj, args = _hexel_inputs(model, camera, near, scale_modifier, conservative_bounds)
    out = rasterize_hexels(
        *args,
        color_model=model.color_index,
        tile_size=tile_size,
        near=near,
        background=background,
        return_contrib=return_contrib,
        abs_grad_accum=abs_grad_accum,
        pixel_count_accum=pixel_count_accum,
        xy_grad_accum=xy_grad_accum,
    )
    if not return_aux:
        return out.image
    return RenderHexelsAux(
        *out,
        means2d=proj.means2d,
        valid=proj.valid,
        projection=proj,
    )


@torch.no_grad()
def render_hexels_owner(
    model: HexelModel,
    camera: Camera,
    background: torch.Tensor | None = None,
    near: float = NEAR_PLANE,
    scale_modifier: float = 1.0,
    tile_size: int = DEFAULT_TILE_SIZE,
    conservative_bounds: bool = False,
) -> HexelOwner:
    """Per-pixel maximum-compositing-weight primitive (id, weight, local
    (u, v), SDF value, depth), hexels' `render_owner`, used for
    contact-patch and wireframe extraction."""
    _, args = _hexel_inputs(model, camera, near, scale_modifier, conservative_bounds)
    return rasterize_hexels_owner(
        *args,
        color_model=model.color_index,
        tile_size=tile_size,
        near=near,
        background=background,
    )

"""torch.autograd.Function wrapping the Metal tile-based Hexel rasterizer
(metalsplat.kernels.rasterize_hexels). Forward/backward math matches
metalsplat.reference.hexels_ref.rasterize_hexels -- validated against it in
tests/test_rasterize_hexels.py.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.ops._validate import (
    check_accumulator,
    check_float32,
    check_shape,
    kernel_background,
    require_mps,
)
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE, bin_and_sort_rects
from metalsplat.reference.hexels_ref import NEAR_PLANE

# Packed channel counts; must match rasterize_hexels.metal.
_F_COUNT = 15
_I_COUNT = 3
_G_COUNT = 11


class HexelRaster(NamedTuple):
    """Rasterizer outputs. `image`, `alpha`, `depth`, `normal`,
    `distortion`, `median_depth` and `overdraw` are differentiable;
    `overdraw_hits`, `final_T` and `contrib` are forward-only.

    `depth` and `normal` are alpha-weighted *sums* (divide by `alpha` to
    get the expected value), matching hexel-kernel and metalsplat's 2DGS.
    `contrib` is each primitive's summed compositing weight over the image
    (zeros unless requested), the statistic hexels' MCMC relocation uses.
    """

    image: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W)
    depth: torch.Tensor  # (H, W)
    normal: torch.Tensor  # (H, W, 3)
    distortion: torch.Tensor  # (H, W)
    median_depth: torch.Tensor  # (H, W)
    overdraw: torch.Tensor  # (H, W)
    overdraw_hits: torch.Tensor  # (H, W)
    final_T: torch.Tensor  # (H, W)
    contrib: torch.Tensor  # (N,)


class HexelOwner(NamedTuple):
    """Maximum-compositing-weight primitive per pixel (forward only)."""

    image: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W)
    owner_id: torch.Tensor  # (H, W) int64, -1 where nothing composited
    owner_weight: torch.Tensor  # (H, W)
    owner_uv: torch.Tensor  # (H, W, 2)
    owner_sdf: torch.Tensor  # (H, W)
    owner_depth: torch.Tensor  # (H, W)


def _params(
    tiles_x, width, height, tile_size, color_model, contrib, owner, n, near, bg, device
):
    ip = torch.tensor(
        [tiles_x, width, height, tile_size, color_model, int(contrib), int(owner), n],
        dtype=torch.int32,
        device=device,
    )
    fp = torch.cat([torch.tensor([near], dtype=torch.float32, device=device), bg])
    return ip, fp


def _launch_forward(
    inputs,
    sorted_ids,
    tile_bins,
    tiles_x,
    tiles_y,
    width,
    height,
    tile_size,
    near,
    color_model,
    background,
    return_contrib,
    return_owner,
):
    transform, opacities, softness, colors, cgrads, cquads, normals = inputs
    device = transform.device
    n = transform.shape[0]
    out_f = torch.zeros(height, width, _F_COUNT, device=device, dtype=torch.float32)
    out_f[..., 14] = 1.0  # final T for pixels no tile covers
    out_i = torch.full((height, width, _I_COUNT), -1, device=device, dtype=torch.int32)
    contrib = torch.zeros(
        max(n, 1) if return_contrib else 1, device=device, dtype=torch.float32
    )
    owner_f = torch.zeros(
        (height * width * 5) if return_owner else 1, device=device, dtype=torch.float32
    )
    owner_i = torch.full(
        ((height * width) if return_owner else 1,), -1, device=device, dtype=torch.int32
    )
    ip, fp = _params(
        tiles_x,
        width,
        height,
        tile_size,
        color_model,
        return_contrib,
        return_owner,
        n,
        near,
        background,
        device,
    )
    if n > 0 and tiles_x > 0 and tiles_y > 0:
        load_kernel("rasterize_hexels").rasterize_hexels_forward(
            transform,
            opacities,
            softness,
            colors,
            cgrads,
            cquads,
            normals,
            sorted_ids,
            tile_bins,
            ip,
            fp,
            out_f,
            out_i,
            contrib,
            owner_f,
            owner_i,
            threads=(tiles_x * tile_size, tiles_y * tile_size),
            group_size=(tile_size, tile_size),
        )
    return out_f, out_i, contrib[:n], owner_f, owner_i, ip, fp


class _RasterizeHexels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        transform,  # (N, 9)
        opacities,  # (N,)
        softness,  # (N, 3)
        colors,  # (N, 3)
        cgrads,  # (N, 6)
        cquads,  # (N, 9)
        normals,  # (N, 3)
        depths,  # (N,) center depth (not differentiated)
        sorted_ids,
        tile_bins,
        tiles_x: int,
        tiles_y: int,
        width: int,
        height: int,
        tile_size: int,
        near: float,
        color_model: int,
        background,
        return_contrib: bool,
        abs_accum,
        pixel_accum,
        xy_accum,
    ):
        inputs = tuple(
            t.contiguous()
            for t in (transform, opacities, softness, colors, cgrads, cquads, normals)
        )
        sorted_ids = sorted_ids.to(torch.int32).contiguous()
        if sorted_ids.numel() == 0:
            sorted_ids = torch.zeros(1, dtype=torch.int32, device=transform.device)
        tile_bins = tile_bins.to(torch.int32).contiguous()
        out_f, out_i, contrib, _, _, ip, fp = _launch_forward(
            inputs,
            sorted_ids,
            tile_bins,
            tiles_x,
            tiles_y,
            width,
            height,
            tile_size,
            near,
            color_model,
            background,
            return_contrib,
            False,
        )
        ctx.save_for_backward(
            *inputs,
            depths.detach().contiguous(),
            sorted_ids,
            tile_bins,
            ip,
            fp,
            out_f,
            out_i,
        )
        ctx.dims = (tiles_x, tiles_y, tile_size)
        ctx.accums = (abs_accum, pixel_accum, xy_accum)
        ctx.color_model = color_model

        image = out_f[..., 0:3].clone()
        alpha = out_f[..., 3].clone()
        depth = out_f[..., 4].clone()
        normal = out_f[..., 5:8].clone()
        distortion = out_f[..., 8].clone()
        median = out_f[..., 9].clone()
        overdraw = out_f[..., 10].clone()
        hits = out_f[..., 11].clone()
        final_T = out_f[..., 14].clone()
        ctx.mark_non_differentiable(hits, final_T, contrib)
        return (
            image,
            alpha,
            depth,
            normal,
            distortion,
            median,
            overdraw,
            hits,
            final_T,
            contrib,
        )

    @staticmethod
    def backward(
        ctx, g_image, g_alpha, g_depth, g_normal, g_dist, g_median, g_overdraw, *_
    ):
        (
            transform,
            opacities,
            softness,
            colors,
            cgrads,
            cquads,
            normals,
            depths,
            sorted_ids,
            tile_bins,
            ip,
            fp,
            out_f,
            out_i,
        ) = ctx.saved_tensors
        tiles_x, tiles_y, tile_size = ctx.dims
        device = transform.device
        n = transform.shape[0]
        height, width = out_f.shape[0], out_f.shape[1]

        def grad_or_zero(g, shape):
            return g if g is not None else torch.zeros(shape, device=device)

        grad_f = (
            torch.cat(
                [
                    grad_or_zero(g_image, (height, width, 3)),
                    grad_or_zero(g_alpha, (height, width))[..., None],
                    grad_or_zero(g_depth, (height, width))[..., None],
                    grad_or_zero(g_normal, (height, width, 3)),
                    grad_or_zero(g_dist, (height, width))[..., None],
                    grad_or_zero(g_median, (height, width))[..., None],
                    grad_or_zero(g_overdraw, (height, width))[..., None],
                ],
                dim=-1,
            )
            .to(torch.float32)
            .contiguous()
        )
        assert grad_f.shape[-1] == _G_COUNT

        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)

        d_transform, d_op, d_soft = zeros(n, 9), zeros(n), zeros(n, 3)
        d_colors, d_cgrads, d_cquads, d_normals = (
            zeros(n, 3),
            zeros(n, 6),
            zeros(n, 9),
            zeros(n, 3),
        )
        abs_accum, pixel_accum, xy_accum = ctx.accums
        want_accum = abs_accum is not None or pixel_accum is not None
        abs_accum = zeros(n) if abs_accum is None else abs_accum
        pixel_accum = zeros(n) if pixel_accum is None else pixel_accum
        # Backward reuses the forward's int params with its own flags in
        # slots 5/6: reduce normal gradients / densification accumulators
        # only when something consumes them.
        bip = ip.clone()
        bip[5] = int(g_normal is not None)
        bip[6] = int(want_accum)

        if n > 0 and tiles_x > 0 and tiles_y > 0:
            load_kernel("rasterize_hexels").rasterize_hexels_backward(
                transform,
                opacities,
                softness,
                colors,
                cgrads,
                cquads,
                normals,
                depths,
                sorted_ids,
                tile_bins,
                bip,
                fp,
                out_f,
                out_i,
                grad_f,
                d_transform,
                d_op,
                d_soft,
                d_colors,
                d_cgrads,
                d_cquads,
                d_normals,
                abs_accum,
                pixel_accum,
                threads=(tiles_x * tile_size, tiles_y * tile_size),
                group_size=(tile_size, tile_size),
            )
        if xy_accum is not None:
            # Signed screen gradient: row0[2] = means2d.x * z_mean exactly, so
            # d(means2d) at fixed depth is the transform's mean-column
            # gradient times the center depth (summed over pixels, linear).
            xy_accum += d_transform[:, [2, 5]] * depths[:, None]
        return (
            d_transform,
            d_op,
            d_soft,
            d_colors,
            d_cgrads,
            d_cquads,
            d_normals,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _check_inputs(
    transform,
    opacities,
    softness,
    colors,
    color_grads,
    color_quads,
    normals,
    depths,
    rects,
    valid,
    tile_size,
):
    device = transform.device
    n = transform.shape[0]
    require_mps("transform", device)
    check_float32("transform", transform, (n, 9), device)
    check_float32("opacities", opacities, (n,), device)
    check_float32("softness", softness, (n, 3), device)
    check_float32("colors", colors, (n, 3), device)
    check_float32("color_grads", color_grads, (n, 6), device)
    check_float32("color_quads", color_quads, (n, 9), device)
    check_float32("normals", normals, (n, 3), device)
    check_float32("depths", depths, (n,), device)
    check_shape("rects", rects, (n, 4))
    check_shape("valid", valid, (n,))
    if (tile_size * tile_size) % 32 != 0 or tile_size > 16:
        raise ValueError(
            "tile_size^2 must be a multiple of the 32-lane SIMD width and "
            "tile_size at most 16 (the kernels reduce per SIMD group within "
            f"one tile, staging up to 8 groups), got {tile_size}"
        )


def rasterize_hexels(
    transform: torch.Tensor,
    opacities: torch.Tensor,
    softness: torch.Tensor,
    colors: torch.Tensor,
    color_grads: torch.Tensor,
    color_quads: torch.Tensor,
    normals: torch.Tensor,
    depths: torch.Tensor,
    rects: torch.Tensor,
    valid: torch.Tensor,
    img_width: int,
    img_height: int,
    color_model: int = 1,
    tile_size: int = DEFAULT_TILE_SIZE,
    near: float = NEAR_PLANE,
    background: torch.Tensor | None = None,
    return_contrib: bool = False,
    abs_grad_accum: torch.Tensor | None = None,
    pixel_count_accum: torch.Tensor | None = None,
    xy_grad_accum: torch.Tensor | None = None,
) -> HexelRaster:
    """Tile-based differentiable Hexel rasterization on MPS.

    Per-primitive inputs come from `reference.hexels_ref.project_hexels`
    (transform (N, 9) flattened row-major, camera-facing normals, center
    depths, rects/valid) and the model (opacity, per-edge softness (N, 3),
    base color with any view residual folded in, color_grads (N, 6) as
    [g_u, g_v], color_quads (N, 9) as [q_uu, q_uv, q_vv]). `color_model`
    is the index into `hexels_ref.COLOR_MODELS`.

    The accumulators, if given, are updated in place by backward() --
    `abs_grad_accum` (N,) with each primitive's summed per-pixel
    |d loss / d screen center| (the AbsGS-style densification signal, in
    pixels), `pixel_count_accum` (N,) with the number of pixels it
    composited into, `xy_grad_accum` (N, 2) with the signed screen-space
    gradient (hexels' `viewspace_points.grad[:, :2]` up to a units factor:
    pixels here, CUDA NDC-scaled pixels there).
    """
    _check_inputs(
        transform,
        opacities,
        softness,
        colors,
        color_grads,
        color_quads,
        normals,
        depths,
        rects,
        valid,
        tile_size,
    )
    device = transform.device
    n = transform.shape[0]
    check_accumulator("abs_grad_accum", abs_grad_accum, n, device)
    check_accumulator("pixel_count_accum", pixel_count_accum, n, device)
    if xy_grad_accum is not None:
        check_float32("xy_grad_accum", xy_grad_accum, (n, 2), device)
        if not xy_grad_accum.is_contiguous():
            raise ValueError("xy_grad_accum is updated in place and must be contiguous")
    background = kernel_background(background, device)

    binning = bin_and_sort_rects(
        rects.detach(), depths.detach(), valid, img_width, img_height, tile_size
    )
    out = _RasterizeHexels.apply(
        transform,
        opacities,
        softness,
        colors,
        color_grads,
        color_quads,
        normals,
        depths,
        binning.sorted_gaussian_ids,
        binning.tile_bins,
        binning.tiles_x,
        binning.tiles_y,
        img_width,
        img_height,
        tile_size,
        near,
        color_model,
        background,
        return_contrib,
        abs_grad_accum,
        pixel_count_accum,
        xy_grad_accum,
    )
    return HexelRaster(*out)


@torch.no_grad()
def rasterize_hexels_owner(
    transform: torch.Tensor,
    opacities: torch.Tensor,
    softness: torch.Tensor,
    colors: torch.Tensor,
    color_grads: torch.Tensor,
    color_quads: torch.Tensor,
    normals: torch.Tensor,
    depths: torch.Tensor,
    rects: torch.Tensor,
    valid: torch.Tensor,
    img_width: int,
    img_height: int,
    color_model: int = 1,
    tile_size: int = DEFAULT_TILE_SIZE,
    near: float = NEAR_PLANE,
    background: torch.Tensor | None = None,
) -> HexelOwner:
    """The per-pixel maximum-weight ("owner") primitive and its local
    coordinates, SDF value and depth -- hexel-kernel's `render_owner`."""
    _check_inputs(
        transform,
        opacities,
        softness,
        colors,
        color_grads,
        color_quads,
        normals,
        depths,
        rects,
        valid,
        tile_size,
    )
    device = transform.device
    background = kernel_background(background, device)
    binning = bin_and_sort_rects(rects, depths, valid, img_width, img_height, tile_size)
    sorted_ids = binning.sorted_gaussian_ids.to(torch.int32).contiguous()
    if sorted_ids.numel() == 0:
        sorted_ids = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = tuple(
        t.detach().contiguous()
        for t in (
            transform,
            opacities,
            softness,
            colors,
            color_grads,
            color_quads,
            normals,
        )
    )
    out_f, _, _, owner_f, owner_i, _, _ = _launch_forward(
        inputs,
        sorted_ids,
        binning.tile_bins.to(torch.int32).contiguous(),
        binning.tiles_x,
        binning.tiles_y,
        img_width,
        img_height,
        tile_size,
        near,
        color_model,
        background,
        False,
        True,
    )
    owner_f = owner_f.reshape(img_height, img_width, 5)
    return HexelOwner(
        image=out_f[..., 0:3].clone(),
        alpha=out_f[..., 3].clone(),
        owner_id=owner_i.reshape(img_height, img_width).to(torch.int64),
        owner_weight=owner_f[..., 0].clone(),
        owner_uv=owner_f[..., 1:3].clone(),
        owner_sdf=owner_f[..., 3].clone(),
        owner_depth=owner_f[..., 4].clone(),
    )

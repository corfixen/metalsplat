"""Checks on the pure-PyTorch Hexel reference itself (CPU-only)."""

import math

import pytest
import torch

from metalsplat.reference import hexels_ref as H
from metalsplat.utils.quaternion import quat_to_rotmat

FX = FY = 40.0
CX, CY = 20.0, 18.0
W, HT = 40, 36


def _scene(n, seed=0, soft_range=(2.0, 12.0), aniso=True):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g) * 2.0 + 2.5
    x = (torch.rand(n, generator=g) * 2 - 1) * 0.4 * z
    y = (torch.rand(n, generator=g) * 2 - 1) * 0.4 * z
    means = torch.stack([x, y, z], -1)
    scales = torch.rand(n, 2, generator=g) * 0.3 + 0.1
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opacities = torch.rand(n, generator=g) * 0.7 + 0.25
    lo, hi = soft_range
    soft = torch.rand(n, 3 if aniso else 1, generator=g) * (hi - lo) + lo
    soft = soft.expand(n, 3).contiguous()
    return means, scales, quats, opacities, soft


def _project(means, scales, quats, opacities, soft, **kw):
    return H.project_hexels(
        means, scales, quats, opacities, soft, torch.eye(3), torch.zeros(3),
        FX, FY, CX, CY, W, HT, **kw,
    )  # fmt: skip


def test_hex_sdf_geometry():
    # Center is one apothem inside; edge midpoints and vertices are on it.
    d, _ = H.hex_sdf_edge(torch.tensor([0.0]), torch.tensor([0.0]))
    assert d.item() == pytest.approx(-1.0)
    r = H.HEX_RADIUS
    for ang in range(6):
        a = math.radians(60 * ang)
        d, _ = H.hex_sdf_edge(
            torch.tensor([r * math.cos(a)]), torch.tensor([r * math.sin(a)])
        )
        assert d.item() == pytest.approx(0.0, abs=1e-6), ang
    d, e = H.hex_sdf_edge(torch.tensor([0.0]), torch.tensor([1.0]))
    assert d.item() == pytest.approx(0.0) and e.item() == 0
    # Scaling a point scales (sdf + 1): the SDF is a gauge of the hexagon.
    u, v = torch.randn(100), torch.randn(100)
    d1, _ = H.hex_sdf_edge(u, v)
    d2, _ = H.hex_sdf_edge(2 * u, 2 * v)
    torch.testing.assert_close(d2 + 1, 2 * (d1 + 1))


def test_hex_sdf_tie_breaking_matches_cuda():
    # Slant tie (v = 0, both slants equal): CUDA picks the second slant.
    _, e = H.hex_sdf_edge(torch.tensor([0.5]), torch.tensor([0.0]))
    assert e.item() == 2
    # Slant vs horizontal tie at the vertex (r/2, 1): a slant wins.
    _, e = H.hex_sdf_edge(torch.tensor([H.HEX_RADIUS / 2]), torch.tensor([1.0]))
    assert e.item() in (1, 2)


def test_bounded_coordinates():
    u, v = torch.tensor([0.3, 2.0]), torch.tensor([-0.2, 0.0])
    ub, vb = H.bounded_quadratic_coordinates(u, v)
    assert ub[0] == u[0] and vb[0] == v[0]  # inside: identity
    d, _ = H.hex_sdf_edge(ub[1:], vb[1:])
    assert d.item() == pytest.approx(0.0, abs=1e-6)  # outside: on the boundary


def test_uv_solve_matches_world_space_ray_plane_intersection():
    """The homogeneous (u, v) solve equals the direct ray-plane
    intersection Hexel Edge's shader uses: depth = n.(c - o) / n.d along an
    unnormalized camera ray d with d.z = 1, then u = (p - c).t_u / s_u."""
    means, scales, quats, opacities, soft = _scene(5, seed=3)
    proj = _project(means, scales, quats, opacities, soft)
    R = quat_to_rotmat(quats)
    for i in range(5):
        for px, py in [(7.5, 9.5), (20.5, 18.5), (31.5, 4.5)]:
            row0, row1, row2 = proj.transform[i]
            hu, hv = row0 - px * row2, row1 - py * row2
            cr = torch.linalg.cross(hu, hv)
            u_h, v_h = cr[0] / cr[2], cr[1] / cr[2]
            z_h = row2[0] * u_h + row2[1] * v_h + row2[2]

            ray = torch.tensor([(px - CX) / FX, (py - CY) / FY, 1.0])
            n = R[i, :, 2]
            t = (n @ means[i]) / (n @ ray)
            p = t * ray - means[i]
            u_w = (p @ R[i, :, 0]) / scales[i, 0]
            v_w = (p @ R[i, :, 1]) / scales[i, 1]
            torch.testing.assert_close(u_h, u_w, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(v_h, v_w, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(z_h, t, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("seed", range(4))
def test_rects_cover_every_contributing_pixel(seed):
    """With conservative bounds, the tile rectangle bounds every pixel where
    a primitive's alpha reaches the 1/255 cutoff (in front of the near
    plane), so binning can never drop a contribution. (The default legacy
    window deliberately can, matching hexel-kernel.)"""
    n = 10
    means, scales, quats, opacities, soft = _scene(n, seed=seed, soft_range=(1.0, 20.0))
    proj = _project(means, scales, quats, opacities, soft, conservative_bounds=True)
    ys, xs = torch.meshgrid(
        torch.arange(HT) + 0.5, torch.arange(W) + 0.5, indexing="ij"
    )
    px, py = xs.reshape(-1), ys.reshape(-1)
    for i in range(n):
        row0, row1, row2 = proj.transform[i]
        hu = row0[None] - px[:, None] * row2[None]
        hv = row1[None] - py[:, None] * row2[None]
        cr = torch.linalg.cross(hu, hv)
        u, v = cr[:, 0] / cr[:, 2], cr[:, 1] / cr[:, 2]
        z = row2[0] * u + row2[1] * v + row2[2]
        d, e = H.hex_sdf_edge(u, v)
        alpha = (opacities[i] * torch.sigmoid(-d * soft[i][e])).clamp_max(H.ALPHA_MAX)
        hit = (alpha >= H.ALPHA_MIN) & (z >= H.NEAR_PLANE)
        if not bool(hit.any()):
            continue
        assert bool(proj.valid[i])
        x0, y0, x1, y1 = proj.rects[i].tolist()
        assert bool((px[hit] >= x0).all() and (px[hit] <= x1).all()), i
        assert bool((py[hit] >= y0).all() and (py[hit] <= y1).all()), i


def test_rects_are_tight():
    """A face-on hexel's rect is its support hexagon plus the 1 px pad, not
    a loose circle: the width is 2 * circumradius, the height 2 * apothem."""
    means = torch.tensor([[0.0, 0.0, 4.0]])
    scales = torch.tensor([[0.5, 0.5]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    op = torch.tensor([0.9])
    soft = torch.full((1, 3), 10.0)
    proj = _project(means, scales, quats, op, soft, conservative_bounds=True)
    a = H.support_apothem(op, soft).item()
    px_per_unit = FX * 0.5 / 4.0
    x0, y0, x1, y1 = proj.rects[0].tolist()
    assert (x1 - x0) == pytest.approx(2 * H.HEX_RADIUS * a * px_per_unit + 2, rel=1e-4)
    assert (y1 - y0) == pytest.approx(2 * a * px_per_unit + 2, rel=1e-4)


def test_culling():
    means = torch.tensor(
        [[0.0, 0.0, 0.1], [0.0, 0.0, 3.0], [50.0, 0.0, 3.0], [0.0, 0.0, 3.0]]
    )
    scales = torch.full((4, 2), 0.2)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 4)
    op = torch.tensor([0.9, 0.9, 0.9, 0.001])
    soft = torch.full((4, 3), 10.0)
    proj = _project(means, scales, quats, op, soft)
    # behind near, visible, off-screen, below the alpha cutoff everywhere
    assert proj.valid.tolist() == [False, True, False, False]


def test_normals_face_camera():
    means, scales, quats, opacities, soft = _scene(20, seed=1)
    proj = _project(means, scales, quats, opacities, soft)
    assert bool(((proj.normal * proj.view_dirs).sum(-1) <= 0).all())
    torch.testing.assert_close(proj.raw_normal, quat_to_rotmat(quats)[..., 2])


def test_view_residuals():
    n = 6
    g = torch.Generator().manual_seed(0)
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    normals = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    assert H.view_color_residual("none", dirs, normals, None, None, None) is None
    vg = torch.randn(n, 3, 3, generator=g)
    lin = H.view_color_residual("linear", dirs, normals, vg, None, None)
    torch.testing.assert_close(lin[:, 1], (vg[:, 1, :] * dirs).sum(-1))
    sh = torch.randn(n, 15, 3, generator=g)
    r0 = H.view_color_residual("sh3", dirs, normals, sh, None, None, sh_degree=0)
    assert bool((r0 == 0).all())
    r3 = H.view_color_residual("sh3", dirs, normals, sh, None, None, sh_degree=3)
    torch.testing.assert_close(r3, H.eval_sh3_residual(sh, dirs, 3))
    ks = torch.rand(n, 3, generator=g)
    log_s = torch.rand(n, generator=g)
    lobe = H.view_color_residual("lobe", dirs, normals, None, ks, log_s)
    facing = (normals * dirs).sum(-1) < 0
    assert bool((lobe[~facing] == 0).all())
    cos = -(normals * dirs).sum(-1)
    expected = ks * (cos.clamp_min(0) ** log_s.clamp(0, 5).exp())[:, None]
    torch.testing.assert_close(lobe, expected)


def test_raster_basic_invariants():
    means, scales, quats, opacities, soft = _scene(8, seed=2)
    proj = _project(means, scales, quats, opacities, soft)
    n = 8
    colors = torch.rand(n, 3)
    out = H.rasterize_hexels(
        proj.transform, opacities, soft, colors, torch.zeros(n, 6), torch.zeros(n, 9),
        proj.normal, proj.depths, proj.valid, W, HT, color_model=0,
        background=torch.tensor([0.2, 0.3, 0.4]),
    )  # fmt: skip
    assert bool((out.alpha >= 0).all() and (out.alpha <= 1).all())
    # Flat color: the image is a convex combination of colors and background.
    assert bool((out.image >= 0).all() and (out.image <= 1).all())
    # Contrib sums the compositing weights, which sum to alpha.
    torch.testing.assert_close(out.contrib.sum(), out.alpha.sum(), rtol=1e-4, atol=1e-4)
    # Owner id is -1 exactly where nothing composited.
    assert bool(((out.owner_id < 0) == (out.alpha == 0)).all())
    assert bool((out.overdraw_hits >= (out.alpha > 0).float()).all())


def test_legacy_window_only_clips():
    """The default bounds are the exact ones intersected with hexel-kernel's
    historical window: never larger, and identical for small primitives."""
    means, scales, quats, opacities, soft = _scene(40, seed=7, soft_range=(1.0, 20.0))
    scales[:5] *= 8.0  # a few large sheets the window clips
    exact = _project(means, scales, quats, opacities, soft, conservative_bounds=True)
    legacy = _project(means, scales, quats, opacities, soft)
    both = legacy.valid
    assert bool((both <= exact.valid).all())
    assert bool((legacy.rects[both, :2] >= exact.rects[both, :2] - 1e-4).all())
    assert bool((legacy.rects[both, 2:] <= exact.rects[both, 2:] + 1e-4).all())


def test_legacy_window_tiles():
    """The window's continuous bounds select whole 16 px tiles."""
    means, scales, quats, opacities, soft = _scene(20, seed=8)
    proj = _project(means, scales, quats, opacities, soft, conservative_bounds=True)
    lo, hi, ok = H.legacy_tile_window(proj.transform, soft, opacities, W, HT)
    assert bool(ok.any())
    torch.testing.assert_close((lo[ok] - 0.5) % 16, torch.zeros_like(lo[ok]))
    torch.testing.assert_close((hi[ok] + 0.5) % 16, torch.zeros_like(hi[ok]))

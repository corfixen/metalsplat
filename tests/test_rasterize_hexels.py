"""The Metal Hexel rasterizer against the pure-PyTorch reference: every
output's forward value and every input's gradient."""

import pytest
import torch

import metalsplat as ms
from metalsplat.ops.rasterize_hexels import rasterize_hexels, rasterize_hexels_owner
from metalsplat.reference import hexels_ref as H

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

FX = FY = 36.0
CX, CY = 20.0, 18.0
W, HT = 40, 36  # not tile multiples: exercises partial edge tiles

OUTPUTS = (
    "image",
    "alpha",
    "depth",
    "normal",
    "distortion",
    "median_depth",
    "overdraw",
)


def _leaves(n, seed, color_model):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g) * 2.0 + 2.5
    x = (torch.rand(n, generator=g) * 2 - 1) * 0.45 * z
    y = (torch.rand(n, generator=g) * 2 - 1) * 0.45 * z
    return {
        "means": torch.stack([x, y, z], -1),
        "scales": torch.rand(n, 2, generator=g) * 0.35 + 0.1,
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.rand(n, generator=g) * 0.7 + 0.25,
        "softness": torch.rand(n, 3, generator=g) * 10 + 2,
        "colors": torch.rand(n, 3, generator=g),
        "color_grads": torch.randn(n, 6, generator=g) * 0.2,
        "color_quads": torch.randn(n, 9, generator=g) * 0.1,
    }


def _run(leaves, device, impl, color_model, weights, return_contrib=True, accums=None):
    p = {k: v.clone().to(device).requires_grad_() for k, v in leaves.items()}
    n = p["means"].shape[0]
    proj = H.project_hexels(
        p["means"], p["scales"], torch.nn.functional.normalize(p["quats"], dim=-1),
        p["opacities"], p["softness"], torch.eye(3, device=device),
        torch.zeros(3, device=device), FX, FY, CX, CY, W, HT,
    )  # fmt: skip
    proj.transform.retain_grad()
    bg = torch.tensor([0.1, 0.2, 0.3], device=device)
    common = (
        p["opacities"],
        p["softness"],
        p["colors"],
        p["color_grads"],
        p["color_quads"],
        proj.normal,
        proj.depths,
    )
    if impl == "ref":
        out = H.rasterize_hexels(
            proj.transform,
            *common,
            proj.valid,
            W,
            HT,
            color_model=color_model,
            background=bg,
        )
    else:
        out = rasterize_hexels(
            proj.transform.reshape(n, 9), *common, proj.rects, proj.valid, W, HT,
            color_model=color_model, background=bg, return_contrib=return_contrib,
            **(accums or {}),
        )  # fmt: skip
    loss = sum((getattr(out, k) * weights[k].to(device)).sum() for k in OUTPUTS)
    loss.backward()
    values = {k: getattr(out, k).detach().cpu() for k in OUTPUTS}
    values["contrib"] = out.contrib.cpu()
    values["overdraw_hits"] = out.overdraw_hits.cpu()
    # Unused color terms (e.g. quads under a linear model) get no gradient.
    grads = {
        k: (v.grad if v.grad is not None else torch.zeros_like(v)).cpu()
        for k, v in p.items()
    }
    return values, grads, proj


def _weights(seed=1):
    g = torch.Generator().manual_seed(seed)
    shapes = {"image": (HT, W, 3), "normal": (HT, W, 3)}
    return {k: torch.randn(*shapes.get(k, (HT, W)), generator=g) for k in OUTPUTS}


@pytest.mark.parametrize("color_model", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 7, 30])
def test_matches_reference(color_model, n):
    leaves = _leaves(n, seed=n + 10 * color_model, color_model=color_model)
    weights = _weights()
    ref_v, ref_g, _ = _run(leaves, "cpu", "ref", color_model, weights)
    mps_v, mps_g, _ = _run(leaves, "mps", "metal", color_model, weights)
    for k in (*OUTPUTS, "contrib", "overdraw_hits"):
        torch.testing.assert_close(mps_v[k], ref_v[k], rtol=1e-4, atol=2e-5, msg=k)
    for k in ref_g:
        scale = ref_g[k].abs().max().clamp_min(1.0)
        torch.testing.assert_close(
            mps_g[k] / scale, ref_g[k] / scale, rtol=1e-3, atol=2e-5, msg=f"grad {k}"
        )


def test_isotropic_softness_gradient_sums_edges():
    """An isotropic model's (N,) softness is broadcast to three edges; its
    gradient is the sum over the edge families, through autograd."""
    model = ms.HexelModel.random(40, bound=0.8, device="mps", softness=6.0)
    model.raw_scales.data[:] = -1.6
    model.raw_opacities.data[:] = 1.0
    cam = ms.Camera.look_at(
        torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), torch.tensor([0.0, -1.0, 0.0]),
        FX, FY, CX, CY, W, HT,
    ).to("mps")  # fmt: skip
    ms.render_hexels(model, cam).sum().backward()
    assert model.log_softness.grad.shape == (40,)
    assert bool(model.log_softness.grad.abs().sum() > 0)


def test_screen_space_accumulators():
    """xy_grad_accum is the transform's mean-column gradient times center
    depth (d loss / d means2d at fixed depth); abs_grad_accum bounds its
    norm, and pixel_count_accum counts covered pixels."""
    n = 12
    leaves = _leaves(n, seed=5, color_model=1)
    weights = _weights()
    acc = {
        "abs_grad_accum": torch.zeros(n, device="mps"),
        "pixel_count_accum": torch.zeros(n, device="mps"),
        "xy_grad_accum": torch.zeros(n, 2, device="mps"),
    }
    _, _, proj = _run(leaves, "mps", "metal", 1, weights, accums=acc)
    g_t = proj.transform.grad.reshape(n, 9)
    expected_xy = (
        torch.stack([g_t[:, 2], g_t[:, 5]], -1) * proj.depths.detach()[:, None]
    )
    torch.testing.assert_close(acc["xy_grad_accum"], expected_xy, rtol=1e-3, atol=1e-4)
    assert bool(
        (acc["abs_grad_accum"] + 1e-5 >= acc["xy_grad_accum"].norm(dim=-1)).all()
    )
    assert bool((acc["pixel_count_accum"][proj.valid.cpu().to("mps")] >= 0).all())
    assert acc["pixel_count_accum"].sum().item() > 0


def test_owner_matches_reference():
    n = 15
    leaves = _leaves(n, seed=9, color_model=1)
    proj = H.project_hexels(
        leaves["means"], leaves["scales"], torch.nn.functional.normalize(leaves["quats"], dim=-1),
        leaves["opacities"], leaves["softness"], torch.eye(3), torch.zeros(3),
        FX, FY, CX, CY, W, HT,
    )  # fmt: skip
    common = (
        leaves["opacities"],
        leaves["softness"],
        leaves["colors"],
        leaves["color_grads"],
        leaves["color_quads"],
        proj.normal,
        proj.depths,
    )
    ref = H.rasterize_hexels(proj.transform, *common, proj.valid, W, HT, color_model=1)
    m = lambda t: t.to("mps")
    own = rasterize_hexels_owner(
        m(proj.transform.reshape(n, 9)), *map(m, common), m(proj.rects), m(proj.valid), W, HT,
        color_model=1,
    )  # fmt: skip
    assert torch.equal(own.owner_id.cpu(), ref.owner_id)
    torch.testing.assert_close(
        own.owner_weight.cpu(), ref.owner_weight, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(own.owner_uv.cpu(), ref.owner_uv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(own.owner_sdf.cpu(), ref.owner_sdf, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        own.owner_depth.cpu(), ref.owner_depth, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(own.image.cpu(), ref.image, rtol=1e-4, atol=1e-5)


def _reference_render(model, cam):
    """render_hexels, but through the pure-PyTorch rasterizer on CPU."""
    model = model.to("cpu")
    cam = cam.to("cpu")
    proj = H.project_hexels(
        model.means, model.scales, model.quats, model.opacities, model.softness3,
        cam.R_wc, cam.t_wc, cam.fx, cam.fy, cam.cx, cam.cy, cam.img_width, cam.img_height,
    )  # fmt: skip
    colors = model.colors
    res = H.view_color_residual(
        model.view_model, proj.view_dirs, proj.raw_normal, getattr(model, "view_grads", None),
        getattr(model, "specular_ks", None), getattr(model, "specular_log_s", None),
        sh_degree=model.active_sh_degree,
    )  # fmt: skip
    if res is not None:
        colors = colors + res
    return H.rasterize_hexels(
        proj.transform, model.opacities, model.softness3, colors, model.color_grads_flat(),
        model.color_quads_flat(), proj.normal, proj.depths, proj.valid,
        cam.img_width, cam.img_height, color_model=model.color_index,
    ).image  # fmt: skip


@pytest.mark.parametrize("view_model", ["none", "linear", "sh3", "lobe"])
def test_render_hexels_view_models(view_model):
    torch.manual_seed(0)
    model = ms.HexelModel.random(25, bound=0.7, view_model=view_model, softness=8.0)
    model.raw_scales.data[:] = -1.5
    model.raw_opacities.data[:] = 0.5
    for name in ("view_grads", "specular_ks", "specular_log_s", "color_grads"):
        if hasattr(model, name):
            getattr(model, name).data.normal_(0, 0.3)
    cam = ms.Camera.look_at(
        torch.tensor([0.4, -0.3, -3.0]), torch.zeros(3), torch.tensor([0.0, -1.0, 0.0]),
        FX, FY, CX, CY, W, HT,
    )  # fmt: skip
    ref = _reference_render(model, cam)
    model_mps = model.to("mps")
    # The brute-force reference has no tiles: compare with exact bounds.
    img = ms.render_hexels(model_mps, cam.to("mps"), conservative_bounds=True)
    torch.testing.assert_close(img.cpu(), ref.detach(), rtol=1e-4, atol=2e-5)


def test_empty_and_fully_culled():
    cam = ms.Camera.identity(FX, FY, CX, CY, W, HT).to("mps")
    bg = torch.tensor([0.3, 0.5, 0.7], device="mps")
    model = ms.HexelModel(torch.tensor([[0.0, 0.0, -5.0]], device="mps"))
    aux = ms.render_hexels(model, cam, background=bg, return_aux=True)
    assert torch.allclose(aux.image, bg.expand(HT, W, 3))
    assert bool((aux.final_T == 1).all() and (aux.alpha == 0).all())
    aux.image.sum().backward()
    assert bool((model.means.grad == 0).all())


@pytest.mark.parametrize("conservative", [False, True])
def test_metal_rects_match_reference(conservative):
    from metalsplat.ops.project_hexels import hexel_rects as metal_rects

    g = torch.Generator().manual_seed(4)
    n = 400
    # Spread centers so some primitives straddle the near plane, sit
    # off-screen or behind the camera, and some are too faint to draw.
    means = torch.randn(n, 3, generator=g) * torch.tensor([2.0, 2.0, 1.5])
    means[:, 2] += 1.0
    scales = torch.rand(n, 2, generator=g) * 1.2 + 0.05
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    op = torch.rand(n, generator=g)
    op[:20] = 0.002
    soft = torch.rand(n, 3, generator=g) * 20 + 0.5
    args = (torch.eye(3), torch.zeros(3), FX, FY, CX, CY, W, HT)
    ref = H.project_hexels(
        means, scales, quats, op, soft, *args, conservative_bounds=conservative
    )
    rects, valid = metal_rects(
        ref.transform.to("mps"), op.to("mps"), soft.to("mps"), ref.depths.to("mps"),
        W, HT, H.NEAR_PLANE, conservative,
    )  # fmt: skip
    assert torch.equal(valid.cpu(), ref.valid)
    assert 0 < int(ref.valid.sum()) < n
    torch.testing.assert_close(rects.cpu(), ref.rects, rtol=1e-4, atol=1e-3)

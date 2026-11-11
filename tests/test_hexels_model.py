"""HexelModel parameterization, row operations, I/O and population control
(CPU-only)."""

import pytest
import torch

from metalsplat import HexelModel
from metalsplat.export_hexels import load_hexels_checkpoint, load_ply, save_ply
from metalsplat.mcmc_hexels import (
    compute_relocation,
    densify_and_prune_hexels,
    mcmc_position_noise,
    mcmc_step,
    relocation_binomials,
    remap_rows,
    update_contrib_ema,
)
from metalsplat.optim import migrate_optimizer_state


def _model(n=20, seed=0, **kw):
    torch.manual_seed(seed)
    m = HexelModel.random(n, **kw)
    for name in m.row_names:
        if name not in ("means", "raw_quats"):
            getattr(m, name).data.normal_(0, 0.3)
    return m


@pytest.mark.parametrize("variant,expected", [("v1", 80), ("v2", 116), ("v3", 260)])
def test_variant_sizes_match_paper(variant, expected):
    m = HexelModel.from_points(torch.randn(10, 3), torch.rand(10, 3), variant=variant)
    assert m.bytes_per_primitive() == expected


def test_rows_follow_configuration():
    assert "color_grads" not in HexelModel.random(3, color_model="flat").row_names
    q = HexelModel.random(3, color_model="quadratic_bounded")
    assert q.color_quads.shape == (3, 3, 3)
    assert HexelModel.random(3, view_model="sh3").view_grads.shape == (3, 15, 3)
    lobe = HexelModel.random(3, view_model="lobe")
    assert lobe.specular_ks.shape == (3, 3) and lobe.specular_log_s.shape == (3,)
    assert HexelModel.random(3, softness_mode="aniso").log_softness.shape == (3, 3)
    with pytest.raises(ValueError):
        HexelModel.random(3, color_model="cubic")


def test_activations():
    m = HexelModel(
        torch.zeros(2, 3),
        scales=torch.tensor([[0.1, 0.2]] * 2),
        opacities=torch.tensor([0.3, 0.7]),
        softness=torch.tensor([5.0, 1e-9]),
    )
    torch.testing.assert_close(m.scales, torch.tensor([[0.1, 0.2]] * 2))
    torch.testing.assert_close(m.opacities, torch.tensor([0.3, 0.7]))
    s3 = m.softness3
    assert s3.shape == (2, 3)
    assert s3[0].tolist() == pytest.approx([5.0] * 3)
    assert s3[1].tolist() == pytest.approx([1e-4] * 3)  # floored like hexel-kernel


def test_row_ops_are_lossless():
    m = _model(view_model="sh3", color_model="quadratic", softness_mode="aniso")
    sub = m.select(torch.tensor([3, 1]))
    torch.testing.assert_close(sub.view_grads, m.view_grads[[3, 1]].detach())
    both = m.cat(sub)
    assert both.num_points == 22
    for k, v in m.raw_rows().items():
        torch.testing.assert_close(both.raw_rows()[k][:20], v)
    assert both.active_sh_degree == m.active_sh_degree
    with pytest.raises(ValueError):
        m.cat(HexelModel.random(2))


def test_ply_round_trip(tmp_path):
    m = _model(
        view_model="lobe", color_model="quadratic_bounded", softness_mode="aniso"
    )
    m.active_sh_degree = 2
    save_ply(m, tmp_path / "h.ply")
    back = load_ply(tmp_path / "h.ply")
    assert (
        back.color_model,
        back.view_model,
        back.softness_mode,
        back.active_sh_degree,
    ) == ("quadratic_bounded", "lobe", "aniso", 2)
    for k, v in m.raw_rows().items():
        torch.testing.assert_close(getattr(back, k).detach(), v)


def _hexels_checkpoint(path, view_basis=3, view_nonzero=False, children=0, n=8):
    g = torch.Generator().manual_seed(0)
    state = {
        "means3d": torch.randn(n, 3, generator=g),
        "log_scales": torch.randn(n, 2, generator=g),
        "rotation_quats": torch.randn(n, 4, generator=g),
        "colors": torch.rand(n, 3, generator=g),
        "color_grads": torch.randn(n, 2, 3, generator=g),
        "color_quads": torch.zeros(n, 3, 3),
        "view_grads": (
            torch.randn(n, view_basis, 3, generator=g)
            if view_nonzero
            else torch.zeros(n, view_basis, 3)
        ),
        "specular_ks": torch.zeros(n, 3),
        "specular_log_s": torch.zeros(n),
        "opacity_logits": torch.randn(n, generator=g),
        "log_softness": torch.full((n,), 2.3),
        "active_mask": torch.ones(n, dtype=torch.bool),
        "child_parent_idx": torch.arange(children),
        "child_sign": torch.ones(children),
        "child_layer": torch.ones(children, dtype=torch.long),
        "child_colors": torch.rand(children, 3, generator=g),
        "child_color_grads": torch.zeros(children, 2, 3),
        "child_color_quads": torch.zeros(children, 3, 3),
        "child_view_grads": torch.zeros(children, view_basis, 3),
        "child_specular_ks": torch.zeros(children, 3),
        "child_specular_log_s": torch.zeros(children),
        "child_opacity_logits": torch.zeros(children),
        "child_log_softness": torch.full((children,), 2.3),
    }
    state["active_mask"][0] = False
    torch.save({"iteration": 30000, "model": state}, path)
    return state


def test_load_hexels_checkpoint_infers_variant(tmp_path):
    s = _hexels_checkpoint(tmp_path / "v1.pt")
    m = load_hexels_checkpoint(tmp_path / "v1.pt")
    assert (m.color_model, m.view_model) == ("linear", "none")
    assert m.num_points == 7  # one inactive row dropped
    torch.testing.assert_close(m.means.detach(), s["means3d"][1:])
    torch.testing.assert_close(m.color_grads.detach(), s["color_grads"][1:])
    _hexels_checkpoint(tmp_path / "v2.pt", view_nonzero=True)
    assert load_hexels_checkpoint(tmp_path / "v2.pt").view_model == "linear"
    _hexels_checkpoint(tmp_path / "v3.pt", view_basis=15)
    assert load_hexels_checkpoint(tmp_path / "v3.pt").view_model == "sh3"
    assert load_hexels_checkpoint(tmp_path / "v2.pt", variant="v1").view_model == "none"


def test_load_hexels_checkpoint_flattens_children(tmp_path):
    s = _hexels_checkpoint(tmp_path / "c.pt", children=3)
    s["active_mask"][:] = True
    torch.save({"model": s}, tmp_path / "c.pt")
    m = load_hexels_checkpoint(
        tmp_path / "c.pt", child_offset_frac=0.5, child_scale_ratio=0.5
    )
    assert m.num_points == 11
    parent_scales = s["log_scales"][:3].exp()
    torch.testing.assert_close(m.scales[8:].detach(), parent_scales * 0.5)
    torch.testing.assert_close(m.colors[8:].detach(), s["child_colors"])


def test_relocation_preserves_single_copy():
    binoms = relocation_binomials("cpu")
    op = torch.tensor([0.3, 0.8])
    sc = torch.tensor([[0.1, 0.2], [0.3, 0.3]])
    new_op, new_sc = compute_relocation(op, sc, torch.tensor([1, 1]), binoms)
    torch.testing.assert_close(new_op, op)
    torch.testing.assert_close(new_sc, sc)
    new_op, _ = compute_relocation(op, sc, torch.tensor([2, 3]), binoms)
    # N copies of opacity o' composite back to the original opacity.
    torch.testing.assert_close(1 - (1 - new_op) ** torch.tensor([2.0, 3.0]), op)


def test_mcmc_step_relocates_and_grows():
    torch.manual_seed(0)
    m = _model(n=40, view_model="linear")
    util = torch.rand(40) * 0.5 + 0.5  # alive, well above the threshold
    util[:5] = 0.0  # dead
    new, stats = mcmc_step(m, util, dead_threshold=0.05, cap_max=60, grow_rate=0.1)
    assert stats.n_relocated == 5 and stats.n_added == 4 and new.num_points == 44
    restarted = stats.source_index == -1
    assert bool(restarted[:5].all()) and bool(restarted[40:].all())
    keep = ~restarted
    for k, v in m.raw_rows().items():
        torch.testing.assert_close(new.raw_rows()[k][keep], v[stats.source_index[keep]])
    # Relocated rows copy a live primitive's appearance.
    live = m.colors.detach()[5:]
    assert all(bool((live == c).all(-1).any()) for c in new.colors.detach()[:5])
    ema = remap_rows(util, stats.source_index)
    assert ema.shape == (44,) and bool((ema[restarted] == 0).all())

    opt_old = torch.optim.Adam(m.parameter_groups())
    sum(p.sum() for p in m.parameters()).backward()
    opt_old.step()
    opt_new = migrate_optimizer_state(
        opt_old, torch.optim.Adam(new.parameter_groups()), stats.source_index
    )
    st = opt_new.state[new.means]
    assert bool((st["exp_avg"][restarted] == 0).all())


def test_mcmc_noise_and_ema():
    m = _model(n=10)
    m.raw_opacities.data[:5] = -10.0  # transparent: noise applies
    m.raw_opacities.data[5:] = 10.0  # opaque: gated off
    before = m.means.detach().clone()
    mcmc_position_noise(m, position_lr=1e-3, noise_lr=5e4)
    moved = (m.means.detach() - before).norm(dim=-1)
    assert bool((moved[:5] > 0).all()) and bool((moved[5:] < 1e-6).all())
    ema = update_contrib_ema(None, torch.ones(4))
    ema = update_contrib_ema(ema, torch.ones(4))
    torch.testing.assert_close(ema, torch.full((4,), 0.19))


def test_densify_applies_to_hexels():
    torch.manual_seed(0)
    m = _model(n=30, view_model="sh3")
    grad = torch.rand(30)
    new, stats = densify_and_prune_hexels(
        m, grad, torch.ones(30), grad_threshold=0.5, prune_opacity_thresh=0.0
    )
    assert isinstance(new, HexelModel) and new.view_model == "sh3"
    assert new.num_points == 30 + stats.n_split + stats.n_cloned - stats.n_pruned

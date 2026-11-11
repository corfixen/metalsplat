"""Population control for Hexels: contribution-driven MCMC relocation and
growth, and adaptive density control.

MCMC is a port of hexels' `mcmc.py` and `Hexels3DTrainer._mcmc_step` /
`_mcmc_noise_step` (itself adapted from 3DGS-MCMC, Kheradmand et al. 2024,
to two tangent scales): primitives whose utility -- an EMA of the
rasterizer's per-primitive `contrib`, or their opacity -- falls below a
fraction of the mean are moved onto copies of live primitives sampled in
proportion to utility, with opacity and scale rescaled (Eq. 9) so the
rendered result is approximately preserved; optionally the population then
grows by `grow_rate` toward `cap_max`.

Unlike the reference, which overwrites rows in place and zeroes Adam state
by hand, every function here returns a new model plus a `source_index`
(see `metalsplat.densify.DensifyStats`): feed it to
`metalsplat.optim.migrate_optimizer_state`, and every relocated, rescaled
or new row starts with zeroed Adam moments, exactly the rows the reference
resets.

Adaptive density control is metalsplat's 2DGS version, which only touches
rows through `raw_rows`/`with_rows` and the tangent frame, so it applies to
`HexelModel` unchanged: `densify_and_prune_hexels`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from metalsplat.densify import prune_low_opacity
from metalsplat.densify2dgs import densify_and_prune_2dgs
from metalsplat.hexels_model import HexelModel
from metalsplat.optim import NEW_GAUSSIAN, _select
from metalsplat.utils.quaternion import quat_to_rotmat

_N_MAX = 51

# Split/clone/prune with tangent-plane split offsets; see densify2dgs.
densify_and_prune_hexels = densify_and_prune_2dgs
prune_low_opacity_hexels = prune_low_opacity


@dataclass
class MCMCStats:
    n_relocated: int
    n_added: int
    n_after: int
    # (n_after,) int64: old row each row came from, NEW_GAUSSIAN (-1) for
    # rows whose optimizer state must restart (relocated targets, their
    # rescaled sources, and new rows). Use with migrate_optimizer_state and
    # `remap_rows`.
    source_index: torch.Tensor


def remap_rows(values: torch.Tensor, source_index: torch.Tensor) -> torch.Tensor:
    """Per-primitive statistics (e.g. a contrib EMA) carried through a
    population change: rows follow `source_index`, restarted rows are 0."""
    return _select(values, source_index)


def relocation_binomials(device, dtype=torch.float32) -> torch.Tensor:
    binoms = torch.zeros((_N_MAX, _N_MAX), dtype=dtype)
    for n in range(_N_MAX):
        for k in range(n + 1):
            binoms[n, k] = float(math.comb(n, k))
    return binoms.to(device)


def compute_relocation(
    opacity_old: torch.Tensor,  # (K,) in (0, 1)
    scales_old: torch.Tensor,  # (K, 2) positive
    copies: torch.Tensor,  # (K,) total copies including the source (>= 1)
    binoms: torch.Tensor,  # relocation_binomials()
) -> tuple[torch.Tensor, torch.Tensor]:
    """3DGS-MCMC Eq. 9 for two scales: the opacity and scale each of N
    copies needs so that N stacked copies match the original."""
    dtype = opacity_old.dtype
    n = copies.to(torch.long).clamp(min=1, max=_N_MAX - 1)
    opacity_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / n.to(dtype))
    p = torch.arange(1, _N_MAX, device=opacity_old.device, dtype=torch.long)
    p_f = p.to(dtype)
    sign = torch.where(p % 2 == 1, 1.0, -1.0).to(dtype)
    coeff_base = sign / torch.sqrt(p_f)
    binom = binoms[n][:, p].to(dtype)
    denom = (binom * coeff_base * (opacity_new.unsqueeze(1) ** p_f)).sum(dim=1)
    coeff = opacity_old / denom.clamp_min(torch.finfo(dtype).eps)
    return opacity_new, scales_old * coeff.unsqueeze(1)


def _rescaled(model, src, copies, binoms):
    opacity, scales = compute_relocation(
        model.opacities.detach()[src], model.scales.detach()[src], copies, binoms
    )
    eps = torch.finfo(opacity.dtype).eps
    opacity = opacity.clamp(0.005, 1.0 - eps)
    p = opacity.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p / (1.0 - p)), torch.log(scales.clamp_min(eps))


def _sample(probs: torch.Tensor, count: int) -> torch.Tensor:
    if float(probs.sum()) <= 0:
        probs = torch.ones_like(probs)
    probs = probs / (probs.sum() + torch.finfo(probs.dtype).eps)
    return torch.multinomial(probs, count, replacement=True)


def _jitter(rows, idx, spawn_jitter):
    if spawn_jitter <= 0 or idx.numel() == 0:
        return
    q = torch.nn.functional.normalize(rows["raw_quats"][idx], dim=-1)
    R = quat_to_rotmat(q)
    s = rows["raw_scales"][idx].exp()
    noise = torch.randn(idx.numel(), 3, device=q.device, dtype=q.dtype)
    offset = (
        R[..., 0] * noise[:, 0:1] * s[:, 0:1]
        + R[..., 1] * noise[:, 1:2] * s[:, 1:2]
        + R[..., 2] * noise[:, 2:3] * 0.5 * (s[:, 0:1] + s[:, 1:2])
    )
    rows["means"][idx] += offset * spawn_jitter


@torch.no_grad()
def mcmc_step(
    model: HexelModel,
    utility: torch.Tensor,  # (N,) contrib EMA (mode "contrib") or ignored (mode "opacity")
    dead_threshold: float = 0.005,
    mode: str = "contrib",
    cap_max: int = 0,
    grow_rate: float = 0.005,
    spawn_jitter: float = 0.0,
    binoms: torch.Tensor | None = None,
) -> tuple[HexelModel, MCMCStats]:
    """One relocation (+ optional growth) round, `_mcmc_step` semantics.

    Dead: utility below `dead_threshold * mean(utility)` (mode "contrib"),
    or opacity at most `dead_threshold` (mode "opacity"). Growth adds
    `grow_rate * N` primitives (capped at `cap_max`; 0 disables growth),
    sampled by utility, each paired with a rescaled source.
    """
    device = model.means.device
    n = model.num_points
    binoms = relocation_binomials(device) if binoms is None else binoms
    if mode == "contrib":
        util = utility
        mean_util = float(util.mean()) if n else 0.0
        dead = (
            util < dead_threshold * mean_util
            if mean_util > 0
            else torch.zeros(n, dtype=torch.bool, device=device)
        )
    elif mode == "opacity":
        util = model.opacities.detach()
        dead = util <= dead_threshold
    else:
        raise ValueError(f"mode must be 'contrib' or 'opacity', got {mode!r}")

    rows = {k: v.clone() for k, v in model.raw_rows().items()}
    source = torch.arange(n, device=device)
    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    n_relocated = 0
    if dead_idx.numel() > 0 and alive_idx.numel() > 0:
        src = alive_idx[_sample(util[alive_idx], dead_idx.numel())]
        copies = torch.bincount(src, minlength=n)[src] + 1
        logit_new, log_scale_new = _rescaled(model, src, copies, binoms)
        for value in rows.values():
            value[dead_idx] = value[src]
        rows["raw_opacities"][dead_idx] = logit_new
        rows["raw_scales"][dead_idx] = log_scale_new
        rows["raw_opacities"][src] = logit_new
        rows["raw_scales"][src] = log_scale_new
        _jitter(rows, dead_idx, spawn_jitter)
        source[dead_idx] = NEW_GAUSSIAN
        source[src] = NEW_GAUSSIAN
        n_relocated = int(dead_idx.numel())

    n_added = 0
    if cap_max > 0:
        target = min(cap_max, int((1.0 + grow_rate) * n))
        num_add = max(0, target - n)
        if num_add > 0:
            probs = model.opacities.detach() if mode == "opacity" else util
            src = _sample(probs, num_add)
            copies = torch.bincount(src, minlength=n)[src] + 1
            current = model.with_rows(rows)
            logit_new, log_scale_new = _rescaled(current, src, copies, binoms)
            new_rows = {k: v[src].clone() for k, v in rows.items()}
            new_rows["raw_opacities"] = logit_new
            new_rows["raw_scales"] = log_scale_new
            rows["raw_opacities"][src] = logit_new
            rows["raw_scales"][src] = log_scale_new
            source[src] = NEW_GAUSSIAN
            _jitter(new_rows, torch.arange(num_add, device=device), spawn_jitter)
            rows = {k: torch.cat([rows[k], new_rows[k]]) for k in rows}
            source = torch.cat(
                [
                    source,
                    torch.full(
                        (num_add,), NEW_GAUSSIAN, device=device, dtype=source.dtype
                    ),
                ]
            )
            n_added = num_add

    if n_relocated == 0 and n_added == 0:
        return model, MCMCStats(0, 0, n, source)
    new_model = model.with_rows(rows)
    return new_model, MCMCStats(n_relocated, n_added, new_model.num_points, source)


@torch.no_grad()
def mcmc_position_noise(
    model: HexelModel,
    position_lr: float,
    noise_lr: float = 5e4,
    normal_noise_frac: float = 0.0,
) -> None:
    """`_mcmc_noise_step`: in-place tangent-plane position noise gated
    toward transparent primitives (sigmoid(100 * ((1 - opacity) - 0.995))),
    scaled by `noise_lr * position_lr` and each axis' squared scale."""
    step = noise_lr * position_lr
    if step == 0.0 or model.num_points == 0:
        return
    opacity = model.opacities.detach()
    gate = torch.sigmoid(100.0 * ((1.0 - opacity) - 0.995))
    s = model.scales.detach()
    R = model.rotmat.detach()
    noise = torch.randn(model.num_points, 3, device=s.device, dtype=s.dtype)
    delta = R[..., 0] * (s[:, 0:1] ** 2 * noise[:, 0:1]) + R[..., 1] * (
        s[:, 1:2] ** 2 * noise[:, 1:2]
    )
    if normal_noise_frac > 0:
        avg = 0.5 * (s[:, 0:1] + s[:, 1:2])
        delta = delta + R[..., 2] * (avg * avg * noise[:, 2:3] * normal_noise_frac)
    model.means.data.add_(delta * (gate * step)[:, None])


def update_contrib_ema(
    ema: torch.Tensor | None, contrib: torch.Tensor, decay: float = 0.9
) -> torch.Tensor:
    """hexels' `_mcmc_update_ema` (decay = `mcmc_ema_decay`)."""
    if ema is None or ema.shape != contrib.shape:
        return contrib.detach().clone() * (1.0 - decay)
    return ema.mul_(decay).add_(contrib.detach(), alpha=1.0 - decay)

"""Trains 3D Hexels on a COLMAP scene on Apple Silicon -- the metalsplat
counterpart of hexels' `scripts.train_scene`.

Defaults reproduce the published NVS runs (the `config.json` shipped with
each pretrained checkpoint, e.g. DTU scan24 V1/V2/V3 fixed-sfm):
- One Hexel per SfM point, identity orientation (`init_normal_mode
  identity`), initial scale 0.25 x the RMS distance to the 3 nearest
  points, capped at 0.05 x the camera extent (`init_scale_mode knn`),
  opacity 0.1, isotropic softness 10.
- Adam with hexels' per-parameter rates; the position rate is a constant,
  unscaled 0.00016 (`spatial_lr_scale False`, no exponential decay), and
  every rate halves at steps 15001, 20001 and 25001
  (`lr_decay_milestones`).
- (1 - 0.2) L1 + 0.2 (1 - SSIM), random training view per step, every
  eighth view held out (`llffhold=8`). Note this is the 3DGS objective;
  `metalsplat.losses.gaussian_splatting_loss` halves the SSIM term.
- `--variant v1|v2|v3`: spatial color; + linear view term; + SH3 view term
  (SH bands grow one per `--sh-interval` steps; 0 = all at once, as hexels'
  default `sh_progressive_interval=0`).
- `--surface`: hexels' surface recipe (configs/surface.json,
  "normal_distortion"): 2DGS-style normal consistency against normals from
  the *median* depth, weight 0.05 from step 7000, and ray distortion,
  weight 1000 from step 3000 ramped over 1000 steps.
- `--mcmc`: contribution-driven relocation every 100 steps between steps
  500 and 15000 (`mcmc_*` defaults), with growth up to `--cap-max`.
- `--densify`: metalsplat's 2DGS adaptive density control between steps
  500 and 15000.
Without `--mcmc`/`--densify` the population is fixed, hexels' default.

Writes held-out renders, a `.ply` of the trained model (reload with
`metalsplat.export_hexels.load_ply`) and per-eval PSNR to `--out`.

Usage:
  uv run python examples/train_hexels.py --source data/dtu/scan24 --variant v3
  uv run python examples/train_hexels.py --source data/garden --images images_4 --surface
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from metalsplat import Camera, HexelModel, render_hexels
from metalsplat.camera import camera_extent
from metalsplat.export_hexels import save_ply
from metalsplat.losses import (
    distortion_loss,
    normal_consistency_loss,
    psnr,
    ssim,
)
from metalsplat.mcmc_hexels import (
    densify_and_prune_hexels,
    mcmc_position_noise,
    mcmc_step,
    relocation_binomials,
    remap_rows,
    update_contrib_ema,
)
from metalsplat.training import GaussianTrainingState, save_image

DEVICE = "mps"


@dataclass
class TrainConfig:
    variant: str = "v1"
    iterations: int = 30_000
    init_scale_mode: str = "knn"  # or "uniform": init_scale_frac x extent
    init_scale_frac: float = 0.002
    init_knn_k: int = 3
    init_knn_multiplier: float = 0.25
    init_knn_max_frac: float = 0.05
    init_random_rotation: bool = False
    init_opacity: float = 0.1
    softness: float = 10.0
    softness_mode: str = "iso"
    lambda_dssim: float = 0.2
    lr_means: float = 0.00016
    spatial_lr_scale: bool = False  # multiply lr_means by the camera extent
    lr_means_final: float = 0.0  # > 0: exponential decay to this (x extent if scaled)
    lr_decay_milestones: tuple = (15_001, 20_001, 25_001)
    lr_decay_factor: float = 0.5
    sh_interval: int = 0
    white_background: bool = False
    # surface recipe
    surface: bool = False
    normal_weight: float = 0.05
    normal_start: int = 7000
    distortion_weight: float = 1000.0
    distortion_start: int = 3000
    distortion_ramp: int = 1000
    # population control
    mcmc: bool = False
    mcmc_interval: int = 100
    mcmc_from: int = 500
    mcmc_until: int = 15_000
    mcmc_dead_threshold: float = 0.005
    mcmc_ema_decay: float = 0.9
    mcmc_noise_lr: float = 5e4
    cap_max: int = 0
    grow_rate: float = 0.005
    densify: bool = False
    densify_interval: int = 100
    densify_from: int = 500
    densify_until: int = 15_000
    max_points: int = 2_000_000
    # logging
    eval_every: int = 1000
    log_every: int = 100
    images_saved: int = 3
    lr_overrides: dict = field(default_factory=dict)


def knn_initial_scales(points, k, multiplier, max_scale, chunk=4096):
    """hexels' `_knn_initial_scales`: multiplier x RMS distance to the k
    nearest other points, capped at `max_scale` (chunked, on device)."""
    out = []
    for i in range(0, points.shape[0], chunk):
        d2 = torch.cdist(points[i : i + chunk], points).square()
        nearest = d2.topk(k + 1, largest=False).values[:, 1:]
        out.append(nearest.mean(dim=1).sqrt())
    scale = torch.cat(out).clamp_min(1e-7**0.5) * multiplier
    if max_scale > 0:
        scale = scale.clamp_max(max_scale)
    return scale


def photometric_loss(pred, target, lambda_dssim):
    """hexels' objective: (1 - lambda) L1 + lambda (1 - SSIM)."""
    l1 = (pred - target).abs().mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(pred, target))


def eval_views(model, cameras, images, background):
    with torch.no_grad():
        values = [
            psnr(render_hexels(model, c, background=background).clamp(0, 1), im).item()
            for c, im in zip(cameras, images)
        ]
    return sum(values) / max(len(values), 1)


def train(
    cfg: TrainConfig,
    train_cameras: list[Camera],
    train_images: list[torch.Tensor],
    points: torch.Tensor,
    colors: torch.Tensor,
    eval_cameras: list[Camera] = (),
    eval_images: list[torch.Tensor] = (),
    out_dir: Path | None = None,
    log=print,
) -> tuple[HexelModel, list[tuple[int, float]]]:
    """Trains and returns the model and its (step, held-out PSNR) history."""
    device = points.device
    extent = camera_extent(train_cameras)
    if cfg.init_scale_mode == "knn":
        scale = knn_initial_scales(
            points,
            cfg.init_knn_k,
            cfg.init_knn_multiplier,
            cfg.init_knn_max_frac * extent,
        )
    else:
        scale = cfg.init_scale_frac * extent
    model = HexelModel.from_points(
        points, colors, variant=cfg.variant, scale=scale,
        opacity=cfg.init_opacity, softness=cfg.softness, softness_mode=cfg.softness_mode,
        random_rotation=cfg.init_random_rotation,
        active_sh_degree=0 if cfg.sh_interval else 3,
    )  # fmt: skip
    lr_scale = extent if cfg.spatial_lr_scale else 1.0
    lr_means0 = cfg.lr_means * lr_scale
    lr_means1 = cfg.lr_means_final * lr_scale
    decay = 1.0  # cumulative milestone factor

    def make_optimizer(m: HexelModel):
        groups = m.parameter_groups({"means": lr_means0, **cfg.lr_overrides})
        for g in groups:
            g["lr"] *= decay
        return torch.optim.Adam(groups)

    state = GaussianTrainingState(model, make_optimizer, track_pixel_count=True)
    background = (
        torch.ones(3, device=device)
        if cfg.white_background
        else torch.zeros(3, device=device)
    )
    binoms = relocation_binomials(device) if cfg.mcmc else None
    contrib_ema = None
    history = []
    s0 = model.scales.detach()[:, 0]
    log(
        f"{model.num_points} Hexels ({cfg.variant}, {model.bytes_per_primitive()} B each), "
        f"extent {extent:.3f}, init scale median {s0.median().item():.5f}"
    )

    def evaluate(step):
        if not eval_cameras:
            return
        value = eval_views(state.model, eval_cameras, eval_images, background)
        history.append((step, value))
        log(f"  eval @ {step}: {value:.2f} dB on {len(eval_cameras)} held-out views")
        if out_dir is not None:
            with torch.no_grad():
                for k in range(min(cfg.images_saved, len(eval_cameras))):
                    aux = render_hexels(
                        state.model,
                        eval_cameras[k],
                        background=background,
                        return_aux=True,
                    )
                    save_image(aux.image, out_dir / f"eval_{k}_step{step}.png")
                    if cfg.surface:
                        n = torch.nn.functional.normalize(
                            aux.normal / aux.alpha.clamp_min(1e-6)[..., None], dim=-1
                        )
                        save_image(
                            (n * 0.5 + 0.5) * (aux.alpha > 0.1)[..., None],
                            out_dir / f"normal_{k}_step{step}.png",
                        )

    evaluate(0)
    start = time.time()
    for step in range(1, cfg.iterations + 1):
        if step in cfg.lr_decay_milestones:
            decay *= cfg.lr_decay_factor
            for group in state.optimizer.param_groups:
                group["lr"] *= cfg.lr_decay_factor
        if lr_means1 > 0:
            t = min(step / cfg.iterations, 1.0)
            lr = decay * math.exp(
                (1 - t) * math.log(lr_means0) + t * math.log(lr_means1)
            )
            for group in state.optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] = lr
        lr = next(g["lr"] for g in state.optimizer.param_groups if g["name"] == "means")

        i = int(torch.randint(len(train_cameras), (1,)))
        cam, target = train_cameras[i], train_images[i]
        need_contrib = cfg.mcmc and cfg.mcmc_from <= step <= cfg.mcmc_until
        aux = render_hexels(
            state.model, cam, background=background, return_aux=True,
            return_contrib=need_contrib, abs_grad_accum=state.grad_accum,
            pixel_count_accum=state.pixel_count,
        )  # fmt: skip
        loss = photometric_loss(aux.image, target, cfg.lambda_dssim)
        if cfg.surface and step >= cfg.distortion_start:
            ramp = min(
                1.0, (step - cfg.distortion_start + 1) / max(cfg.distortion_ramp, 1)
            )
            loss = loss + cfg.distortion_weight * ramp * distortion_loss(aux.distortion)
        if cfg.surface and step >= cfg.normal_start:
            alpha = aux.alpha.detach()
            loss = loss + cfg.normal_weight * normal_consistency_loss(
                aux.normal, aux.median_depth * alpha, alpha, cam
            )
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        state.optimizer.step()
        state.record_visibility(aux.valid)
        if need_contrib:
            contrib_ema = update_contrib_ema(
                contrib_ema, aux.contrib, cfg.mcmc_ema_decay
            )
            mcmc_position_noise(state.model, lr, cfg.mcmc_noise_lr)

        if step % cfg.log_every == 0 or step == 1:
            el = time.time() - start
            log(
                f"step {step:6d}  loss {loss.item():.5f}  n {state.model.num_points}  "
                f"{1000 * el / step:.1f} ms/step"
            )

        if need_contrib and step % cfg.mcmc_interval == 0 and contrib_ema is not None:
            model, stats = mcmc_step(
                state.model, contrib_ema, cfg.mcmc_dead_threshold, cap_max=cfg.cap_max,
                grow_rate=cfg.grow_rate, binoms=binoms,
            )  # fmt: skip
            if model is not state.model:
                contrib_ema = remap_rows(contrib_ema, stats.source_index)
                state.replace_model(model, stats.source_index)
                log(
                    f"  mcmc @ {step}: relocated {stats.n_relocated}, added {stats.n_added} "
                    f"(n={stats.n_after})"
                )
        if (
            cfg.densify
            and cfg.densify_from <= step <= cfg.densify_until
            and step % cfg.densify_interval == 0
        ):
            model, stats = densify_and_prune_hexels(
                state.model, state.grad_accum, state.grad_count, pixel_count=state.pixel_count,
                max_points=cfg.max_points,
            )  # fmt: skip
            state.replace_model(model, stats.source_index, stats.parent_index)
            state.reset_accumulators()
            log(f"  densify @ {step}: {stats.n_before} -> {stats.n_after}")
        if cfg.sh_interval and step % cfg.sh_interval == 0:
            state.model.increase_sh_degree()
        if step % cfg.eval_every == 0:
            evaluate(step)
    if cfg.iterations % cfg.eval_every:
        evaluate(cfg.iterations)
    return state.model, history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="COLMAP scene root")
    ap.add_argument(
        "--images", default="images", help="image subdirectory (e.g. images_4)"
    )
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/hexels"))
    ap.add_argument("--variant", default="v1", choices=("v1", "v2", "v3"))
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--surface", action="store_true")
    ap.add_argument("--mcmc", action="store_true")
    ap.add_argument("--cap-max", type=int, default=0)
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--aniso", action="store_true", help="per-edge softness")
    ap.add_argument("--sh-interval", type=int, default=0)
    ap.add_argument("--white-background", action="store_true")
    ap.add_argument(
        "--all-views", action="store_true", help="no held-out views (surface eval)"
    )
    args = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

    from metalsplat.data.colmap import load_colmap_scene

    scene = load_colmap_scene(
        args.source, image_dir=args.images, device=DEVICE, downscale=args.downscale
    )
    n = len(scene.cameras)
    eval_idx = [] if args.all_views else list(range(0, n, 8))
    train_idx = [i for i in range(n) if i not in set(eval_idx)]
    cfg = TrainConfig(
        variant=args.variant, iterations=args.iterations, surface=args.surface,
        mcmc=args.mcmc, cap_max=args.cap_max, densify=args.densify,
        softness_mode="aniso" if args.aniso else "iso", sh_interval=args.sh_interval,
        white_background=args.white_background,
    )  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    model, history = train(
        cfg,
        [scene.cameras[i] for i in train_idx], [scene.images[i] for i in train_idx],
        scene.points, scene.colors,
        [scene.cameras[i] for i in eval_idx], [scene.images[i] for i in eval_idx],
        out_dir=args.out,
    )  # fmt: skip
    save_ply(model, args.out / "hexels.ply")
    print(f"Done: {model.num_points} Hexels -> {args.out / 'hexels.ply'}")
    if history:
        print(f"Final held-out PSNR {history[-1][1]:.2f} dB")


if __name__ == "__main__":
    main()

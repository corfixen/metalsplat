"""Hexel model I/O: a lossless .ply round trip, and loading checkpoints
trained by the hexels reference trainer (honeycomb-surfels/hexels), so the
published pretrained models render on Apple Silicon.

The .ply stores every raw row as float32 vertex properties
(``means_0..2``, ``raw_scales_0..1``, ..., one property per scalar,
row-major for the matrix-valued rows) and records the configuration in
header comments (``comment hexels color_model=linear``). It is this
package's own interchange format: the Hexel primitive has no established
viewer ecosystem the way 3DGS does, so there is no external layout to
match. hexels' ``hexel_cells.ply`` (polygon cells for surface inspection)
is a different, lossy export.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from metalsplat._ply import read_ply, read_ply_comments, write_ply
from metalsplat.hexels_model import VARIANTS, HexelModel, _row_shapes
from metalsplat.utils.quaternion import quat_to_rotmat

_CONFIG_KEYS = (
    "color_model",
    "view_model",
    "softness_mode",
    "sh_degree",
    "active_sh_degree",
)


def save_ply(model: HexelModel, path: str | Path) -> None:
    columns = {}
    for name in model.row_names:
        value = getattr(model, name).detach().cpu().numpy()
        columns[name] = value.reshape(value.shape[0], -1) if value.ndim > 1 else value
    comments = [f"hexels {k}={getattr(model, k)}" for k in _CONFIG_KEYS]
    write_ply(path, columns, comments=comments)


def load_ply(path: str | Path, device: str | torch.device = "cpu") -> HexelModel:
    """Inverse of `save_ply`."""
    config = {}
    for comment in read_ply_comments(path):
        if comment.startswith("hexels ") and "=" in comment:
            key, value = comment[len("hexels ") :].split("=", 1)
            config[key] = value
    missing = [k for k in _CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"{Path(path).name}: not a Hexel .ply (missing {missing})")
    cols = read_ply(path)
    n = len(next(iter(cols.values())))
    shapes = _row_shapes(
        n, config["color_model"], config["view_model"], config["softness_mode"]
    )
    rows = {}
    for name, shape in shapes.items():
        k = math.prod(shape[1:])
        keys = [name] if len(shape) == 1 else [f"{name}_{i}" for i in range(k)]
        data = np.stack([cols[key] for key in keys], axis=-1)
        rows[name] = torch.from_numpy(data.copy()).reshape(shape).to(device)
    template = HexelModel(
        torch.zeros(0, 3, device=device),
        color_model=config["color_model"],
        view_model=config["view_model"],
        softness_mode=config["softness_mode"],
        sh_degree=int(config["sh_degree"]),
        active_sh_degree=int(config["active_sh_degree"]),
    )
    return template.with_rows(rows)


def _materialize_children(state, child_offset_frac, child_scale_ratio):
    """hexels' `pack_for_rasterizer` children as independent primitives
    (rotation deltas are not supported). Exact for rendering color; note
    hexel-kernel excludes children from its surface outputs, which a
    flattened model cannot."""
    parent = state["child_parent_idx"].long()
    sign = state["child_sign"]
    layer = state.get("child_layer")
    layer = (
        torch.ones_like(sign)
        if layer is None or layer.numel() == 0
        else layer.to(sign.dtype)
    )
    q = torch.nn.functional.normalize(state["rotation_quats"][parent], dim=-1)
    s = state["log_scales"][parent].exp()
    normal = quat_to_rotmat(q)[..., 2]
    offset = child_offset_frac * 0.5 * (s[:, 0] + s[:, 1]) * layer
    return {
        "means3d": state["means3d"][parent] + sign[:, None] * offset[:, None] * normal,
        "log_scales": (s * child_scale_ratio).log(),
        "rotation_quats": q,
        "colors": state["child_colors"],
        "color_grads": state["child_color_grads"],
        "color_quads": state["child_color_quads"],
        "view_grads": state["child_view_grads"],
        "specular_ks": state["child_specular_ks"],
        "specular_log_s": state["child_specular_log_s"],
        "opacity_logits": state["child_opacity_logits"],
        "log_softness": state["child_log_softness"],
    }


def load_hexels_checkpoint(
    path: str | Path,
    variant: str | None = None,
    color_model: str | None = None,
    view_model: str | None = None,
    device: str | torch.device = "cpu",
    child_offset_frac: float = 0.5,
    child_scale_ratio: float = 0.5,
) -> HexelModel:
    """Loads a hexels ``ckpt_*.pt`` (full or compact deployment format)
    into a `HexelModel`.

    hexels checkpoints do not record their appearance configuration, so
    give `variant` ("v1"/"v2"/"v3") or `color_model`/`view_model`. Without
    them it is read from a compact checkpoint's deployment metadata, or
    inferred: 15 view-basis rows mean V3 (SH3); 3 rows mean V2 if any view
    gradient is non-zero, else V1 (V1 training never updates them).

    Inactive base rows (``active_mask``) are dropped; child primitives are
    flattened into independent ones (`child_offset_frac` and
    `child_scale_ratio` must match the training run's options).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = dict(ckpt["model"])
    meta = ckpt.get("deployment") or {}
    if variant is not None:
        color_model, view_model = VARIANTS[variant]
    color_model = color_model or meta.get("color_model") or "linear"
    if view_model is None:
        view_model = meta.get("view_model")
    view_grads = state.get("view_grads")
    if view_model is None:
        if (
            view_grads is not None
            and view_grads.ndim == 3
            and view_grads.shape[1] == 15
        ):
            view_model = "sh3"
        elif (
            view_grads is not None
            and view_grads.numel()
            and bool(view_grads.abs().max() > 0)
        ):
            view_model = "linear"
        elif state.get("specular_ks") is not None and bool(
            state["specular_ks"].abs().max() > 0
        ):
            view_model = "lobe"
        else:
            view_model = "none"

    names = {
        "means": "means3d", "raw_scales": "log_scales", "raw_quats": "rotation_quats",
        "raw_opacities": "opacity_logits", "colors": "colors", "color_grads": "color_grads",
        "color_quads": "color_quads", "view_grads": "view_grads",
        "specular_ks": "specular_ks", "specular_log_s": "specular_log_s",
        "log_softness": "log_softness",
    }  # fmt: skip

    def source_rows(src):
        return {k: src[v].float() for k, v in names.items() if v in src}

    base = source_rows(state)
    active = state.get("active_mask")
    if active is not None and active.numel() == base["means"].shape[0]:
        base = {k: v[active.bool()] for k, v in base.items()}
    parts = [base]
    if "child_parent_idx" in state and state["child_parent_idx"].numel() > 0:
        parts.append(
            source_rows(
                _materialize_children(state, child_offset_frac, child_scale_ratio)
            )
        )

    softness_mode = "aniso" if base["log_softness"].ndim == 2 else "iso"
    n = sum(p["means"].shape[0] for p in parts)
    shapes = _row_shapes(n, color_model, view_model, softness_mode)
    rows = {}
    for name, shape in shapes.items():
        if all(name in p for p in parts):
            value = torch.cat([p[name] for p in parts])
            if value.numel() != math.prod(shape):
                raise ValueError(
                    f"checkpoint {name} has shape {tuple(value.shape)}, but "
                    f"color_model={color_model!r}, view_model={view_model!r} needs "
                    f"{shape}: wrong variant for this checkpoint?"
                )
            rows[name] = value.reshape(shape)
        else:  # compact checkpoints omit inactive appearance rows
            rows[name] = torch.zeros(shape)
    template = HexelModel(
        torch.zeros(0, 3), color_model=color_model, view_model=view_model,
        softness_mode=softness_mode,
    )  # fmt: skip
    return template.with_rows({k: v.to(device) for k, v in rows.items()})

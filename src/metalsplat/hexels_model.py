"""Learnable 3D Hexel (hexagonal surfel) parameters.

A Hexel is a planar primitive like a 2DGS surfel -- center, quaternion
(rotation columns 0/1 are the tangent axes, column 2 the normal) and two
tangent scales -- whose footprint is a soft-edged regular hexagon and whose
color varies across it. See `metalsplat.reference.hexels_ref` for the
rendering model.

Raw parameters follow the hexels reference (`hexels/model.py`) one to one,
under this package's row names so `metalsplat.densify.split_clone_prune`
and `metalsplat.optim` work unchanged:

====================  ==================  ===============================
row                   hexels name         activation
====================  ==================  ===============================
``means``             ``means3d``         --
``raw_scales``        ``log_scales``      exp
``raw_quats``         ``rotation_quats``  normalize (w, x, y, z)
``raw_opacities``     ``opacity_logits``  sigmoid
``colors``            ``colors``          -- (unbounded linear RGB)
``color_grads``       ``color_grads``     -- (N, 2, 3): g_u, g_v
``color_quads``       ``color_quads``     -- (N, 3, 3): q_uu, q_uv, q_vv
``view_grads``        ``view_grads``      -- (N, 3, 3) linear / (N, 15, 3) SH3
``specular_ks``       ``specular_ks``     -- (N, 3), lobe view model
``specular_log_s``    ``specular_log_s``  -- (N,), lobe view model
``log_softness``      ``log_softness``    exp, floored at 1e-4; (N,) or (N, 3)
====================  ==================  ===============================

Only the rows the configuration uses exist: ``color_grads`` for every color
model but ``flat``, ``color_quads`` for the quadratic ones, ``view_grads``
for ``linear``/``sh3`` view models, the specular pair for ``lobe``. The
paper's variants are presets of (color_model, view_model):
V1 = (linear, none), V2 = (linear, linear), V3 = (linear, sh3).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from metalsplat.reference.hexels_ref import (
    COLOR_MODELS,
    SOFTNESS_MIN,
    VIEW_MODELS,
    color_model_index,
)
from metalsplat.sh_color import logit
from metalsplat.utils.quaternion import quat_to_rotmat

VARIANTS = {
    "v1": ("linear", "none"),
    "v2": ("linear", "linear"),
    "v3": ("linear", "sh3"),
}


def _row_shapes(n: int, color_model: str, view_model: str, softness_mode: str) -> dict:
    shapes = {
        "means": (n, 3),
        "raw_scales": (n, 2),
        "raw_quats": (n, 4),
        "raw_opacities": (n,),
        "colors": (n, 3),
    }
    if color_model != "flat":
        shapes["color_grads"] = (n, 2, 3)
    if color_model in ("quadratic", "quadratic_bounded"):
        shapes["color_quads"] = (n, 3, 3)
    if view_model == "linear":
        shapes["view_grads"] = (n, 3, 3)
    elif view_model == "sh3":
        shapes["view_grads"] = (n, 15, 3)
    elif view_model == "lobe":
        shapes["specular_ks"] = (n, 3)
        shapes["specular_log_s"] = (n,)
    shapes["log_softness"] = (n,) if softness_mode == "iso" else (n, 3)
    return shapes


class HexelModel(nn.Module):
    """3D Hexels. Construct with physical values (positive scales, unit
    quaternions, opacities in [0, 1], positive softness), or from a preset
    with `from_points`."""

    def __init__(
        self,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor | None = None,  # (N, 2) positive; default 0.02
        quats: torch.Tensor | None = None,  # (N, 4) (w, x, y, z); default identity
        opacities: torch.Tensor
        | None = None,  # (N,); default 0.1 (hexels init_opacity)
        colors: torch.Tensor | None = None,  # (N, 3); default 0.5 gray
        softness: float | torch.Tensor = 10.0,  # scalar, (N,) or (N, 3)
        color_model: str = "linear",
        view_model: str = "none",
        softness_mode: str = "iso",
        sh_degree: int = 3,
        active_sh_degree: int | None = None,
    ):
        super().__init__()
        n = means.shape[0]
        device, dtype = means.device, means.dtype
        self._configure(
            color_model, view_model, softness_mode, sh_degree, active_sh_degree
        )
        if scales is None:
            scales = torch.full((n, 2), 0.02, device=device, dtype=dtype)
        if quats is None:
            quats = torch.zeros(n, 4, device=device, dtype=dtype)
            quats[:, 0] = 1.0
        if opacities is None:
            opacities = torch.full((n,), 0.1, device=device, dtype=dtype)
        if colors is None:
            colors = torch.full((n, 3), 0.5, device=device, dtype=dtype)
        soft_shape = (n,) if softness_mode == "iso" else (n, 3)
        softness = torch.as_tensor(softness, device=device, dtype=dtype)
        log_softness = torch.broadcast_to(softness, soft_shape).log()

        shapes = _row_shapes(n, color_model, view_model, softness_mode)
        rows = {
            name: torch.zeros(shape, device=device, dtype=dtype)
            for name, shape in shapes.items()
        }
        rows.update(
            means=means,
            raw_scales=scales.log(),
            raw_quats=quats,
            raw_opacities=logit(opacities),
            colors=colors,
            log_softness=log_softness,
        )
        self._set_rows(rows)

    def _configure(
        self, color_model, view_model, softness_mode, sh_degree, active_sh_degree
    ):
        if color_model not in COLOR_MODELS:
            raise ValueError(
                f"color_model must be one of {COLOR_MODELS}, got {color_model!r}"
            )
        if view_model not in VIEW_MODELS:
            raise ValueError(
                f"view_model must be one of {VIEW_MODELS}, got {view_model!r}"
            )
        if softness_mode not in ("iso", "aniso"):
            raise ValueError(
                f"softness_mode must be 'iso' or 'aniso', got {softness_mode!r}"
            )
        if not 0 <= sh_degree <= 3:
            raise ValueError(f"sh_degree must be in [0, 3], got {sh_degree}")
        active = sh_degree if active_sh_degree is None else active_sh_degree
        if not 0 <= active <= sh_degree:
            raise ValueError(
                f"active_sh_degree must be in [0, {sh_degree}], got {active}"
            )
        self.color_model = color_model
        self.view_model = view_model
        self.softness_mode = softness_mode
        self.sh_degree = sh_degree
        self.active_sh_degree = active

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,  # (N, 3)
        colors: torch.Tensor,  # (N, 3) in [0, 1]
        variant: str = "v1",
        scale: float | torch.Tensor = 0.02,
        opacity: float = 0.1,
        softness: float = 10.0,
        softness_mode: str = "iso",
        random_rotation: bool = True,
        active_sh_degree: int | None = None,
    ) -> HexelModel:
        """Hexels seeded at a point cloud, one per point, with a preset
        variant (`VARIANTS`). Random orientations by default, as hexels'
        SfM initialization uses (`init_normal_mode` other than PCA)."""
        if variant not in VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(VARIANTS)}, got {variant!r}"
            )
        color_model, view_model = VARIANTS[variant]
        n = points.shape[0]
        device, dtype = points.device, points.dtype
        # scale: a scalar, an (N,) per-point size, or (N, 2).
        scales = torch.as_tensor(scale, device=device, dtype=dtype)
        if scales.ndim == 1:
            scales = scales[:, None]
        scales = torch.broadcast_to(scales, (n, 2)).clone()
        if random_rotation:
            quats = nn.functional.normalize(
                torch.randn(n, 4, device=device, dtype=dtype), dim=-1
            )
        else:
            quats = None
        return cls(
            points,
            scales=scales,
            quats=quats,
            opacities=torch.full((n,), opacity, device=device, dtype=dtype),
            colors=colors,
            softness=softness,
            color_model=color_model,
            view_model=view_model,
            softness_mode=softness_mode,
            active_sh_degree=0
            if (view_model == "sh3" and active_sh_degree is None)
            else active_sh_degree,
        )

    @classmethod
    def random(
        cls, n: int, bound: float = 1.0, device: str | torch.device = "cpu", **kwargs
    ):
        means = (torch.rand(n, 3, device=device) * 2 - 1) * bound
        return cls(means, colors=torch.rand(n, 3, device=device), **kwargs)

    # ------------------------------------------------------------------ rows

    @property
    def row_names(self) -> tuple[str, ...]:
        return tuple(
            _row_shapes(0, self.color_model, self.view_model, self.softness_mode)
        )

    def _set_rows(self, rows: dict[str, torch.Tensor]) -> None:
        n = rows["means"].shape[0]
        shapes = _row_shapes(n, self.color_model, self.view_model, self.softness_mode)
        if set(rows) != set(shapes):
            raise ValueError(f"expected rows {sorted(shapes)}, got {sorted(rows)}")
        for name, shape in shapes.items():
            if tuple(rows[name].shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(rows[name].shape)}"
                )
        # Registration order is parameters() order: keep it stable.
        for name in shapes:
            setattr(self, name, nn.Parameter(rows[name].detach().clone()))

    def raw_rows(self) -> dict[str, torch.Tensor]:
        """Every per-primitive raw parameter by name, detached (see
        `SplatModel.raw_rows` for why row edits go through these)."""
        return {name: getattr(self, name).detach() for name in self.row_names}

    def with_rows(self, rows: dict[str, torch.Tensor]) -> HexelModel:
        model = type(self).__new__(type(self))
        nn.Module.__init__(model)
        model._configure(
            self.color_model,
            self.view_model,
            self.softness_mode,
            self.sh_degree,
            self.active_sh_degree,
        )
        model._set_rows(rows)
        return model

    def select(self, index: torch.Tensor) -> HexelModel:
        return self.with_rows({k: v[index] for k, v in self.raw_rows().items()})

    def cat(self, *others: HexelModel) -> HexelModel:
        for other in others:
            if (
                type(other) is not type(self)
                or other.row_names != self.row_names
                or other.sh_degree != self.sh_degree
            ):
                raise ValueError(
                    "can only concatenate Hexel models with the same configuration"
                )
        parts = [self.raw_rows()] + [o.raw_rows() for o in others]
        return self.with_rows({k: torch.cat([p[k] for p in parts]) for k in parts[0]})

    @property
    def num_points(self) -> int:
        return self.means.shape[0]

    # ----------------------------------------------------------- activations

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return self.raw_quats / self.raw_quats.norm(dim=-1, keepdim=True).clamp_min(
            1e-8
        )

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacities)

    @property
    def softness3(self) -> torch.Tensor:
        """(N, 3) per-edge softness the rasterizer uses, floored at 1e-4
        (hexel-kernel's `fmaxf(softness, 1e-4)`, so its gradient stops at
        the floor). An isotropic model broadcasts one value to all three."""
        soft = self.log_softness.exp().clamp_min(SOFTNESS_MIN)
        return soft[:, None].expand(-1, 3) if soft.ndim == 1 else soft

    @property
    def rotmat(self) -> torch.Tensor:
        return quat_to_rotmat(self.quats)

    @property
    def normals(self) -> torch.Tensor:
        """(N, 3) unit world-space normal, unoriented."""
        return self.rotmat[..., :, 2]

    @property
    def color_index(self) -> int:
        return color_model_index(self.color_model)

    def color_grads_flat(self) -> torch.Tensor:
        """(N, 6) [g_u, g_v] as the rasterizer reads them (zeros for flat)."""
        if self.color_model == "flat":
            return self.means.new_zeros(self.num_points, 6)
        return self.color_grads.reshape(-1, 6)

    def color_quads_flat(self) -> torch.Tensor:
        """(N, 9) [q_uu, q_uv, q_vv] (zeros for non-quadratic models)."""
        if self.color_model not in ("quadratic", "quadratic_bounded"):
            return self.means.new_zeros(self.num_points, 9)
        return self.color_quads.reshape(-1, 9)

    def increase_sh_degree(self) -> int:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)
        return self.active_sh_degree

    def parameter_groups(self, lrs: dict[str, float] | None = None) -> list[dict]:
        """Adam param groups with hexels' default learning rates
        (`Hexels3DOptions`), overridable per row name. The position rate is
        the unscaled 0.00016: multiply it by the scene extent, as hexels
        does with `spatial_lr_scale`."""
        defaults = {
            "means": 0.00016,
            "raw_scales": 0.005,
            "raw_quats": 0.001,
            "raw_opacities": 0.1,
            "colors": 0.01,
            "color_grads": 0.005,
            "color_quads": 0.005,
            "view_grads": 0.0005 if self.view_model == "sh3" else 0.005,
            "specular_ks": 0.005,
            "specular_log_s": 0.001,
            "log_softness": 0.01,
        }
        defaults.update(lrs or {})
        return [
            {"params": [getattr(self, name)], "lr": defaults[name], "name": name}
            for name in self.row_names
        ]

    def bytes_per_primitive(self) -> int:
        """Learned FP32 state per Hexel (80 / 116 / 260 bytes for V1 / V2 / V3)."""
        return 4 * sum(math.prod(getattr(self, k).shape[1:]) for k in self.row_names)

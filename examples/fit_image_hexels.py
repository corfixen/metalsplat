"""Overfits Hexels to a target image with Adam -- the Hexel counterpart of
`fit_image_2dgs.py`, and an end-to-end smoke test of the Hexel pipeline:
projection -> tile binning -> the Metal hexagon rasterizer, with gradients
reaching every parameter (centers, scales and rotations through the exact
ray-plane intersection; softness through the hexagon boundary; the
spatial color gradients through the per-pixel (u, v)).

Each Hexel carries a first-order color field, so a smooth gradient inside
one shape needs one primitive instead of many -- the image here includes a
shaded disk to exercise that. Saves before/target/after PNGs next to this
script.

Usage: uv run python examples/fit_image_hexels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from metalsplat import Camera, HexelModel, render_hexels
from metalsplat.training import save_image

DEVICE = "mps"
IMG_SIZE = 128
NUM_HEXELS = 600
NUM_ITERS = 400
LR = 0.01


def make_target_image(size: int) -> torch.Tensor:
    img = Image.new("RGB", (size, size), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [size * 0.45, size * 0.35, size * 0.9, size * 0.8], fill=(60, 160, 200)
    )
    draw.ellipse(
        [size * 0.3, size * 0.55, size * 0.75, size * 0.95], fill=(240, 210, 60)
    )
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    # A shaded disk: a linear color ramp, which a single Hexel can represent.
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cx, cy, r = size * 0.4, size * 0.4, size * 0.25
    inside = ((xs - cx) ** 2 + (ys - cy) ** 2) < r * r
    ramp = torch.stack(
        [0.9 - xs / size, 0.2 + 0.5 * ys / size, 0.3 * torch.ones_like(xs)], -1
    )
    arr[inside] = ramp[inside].float()
    return arr


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    target = make_target_image(IMG_SIZE).to(DEVICE)

    # Magnification fx/z == 1: world x/y and scale map 1:1 onto pixels.
    fx = fy = float(IMG_SIZE)
    depth = fx
    torch.manual_seed(0)
    means = torch.zeros(NUM_HEXELS, 3, device=DEVICE)
    means[:, 0] = (torch.rand(NUM_HEXELS, device=DEVICE) - 0.5) * IMG_SIZE * 0.95
    means[:, 1] = (torch.rand(NUM_HEXELS, device=DEVICE) - 0.5) * IMG_SIZE * 0.95
    means[:, 2] = depth + torch.rand(NUM_HEXELS, device=DEVICE)  # distinct depths
    model = HexelModel(
        means,
        scales=torch.full((NUM_HEXELS, 2), 4.0, device=DEVICE),
        opacities=torch.full((NUM_HEXELS,), 0.5, device=DEVICE),
        colors=torch.rand(NUM_HEXELS, 3, device=DEVICE),
        softness=4.0,
        color_model="linear",
    )
    camera = Camera.identity(fx, fy, IMG_SIZE / 2, IMG_SIZE / 2, IMG_SIZE, IMG_SIZE).to(
        DEVICE
    )

    out_dir = Path(__file__).parent
    save_image(target, out_dir / "fit_image_hexels_target.png")
    with torch.no_grad():
        save_image(
            render_hexels(model, camera).clamp(0, 1),
            out_dir / "fit_image_hexels_before.png",
        )

    # Face-on fitting: freeze the rotation (and depth) so primitives stay in
    # the image plane; everything else trains at one rate, as fit_image does.
    model.raw_quats.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    first = None
    for step in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()
        image = render_hexels(model, camera)
        loss = (image - target).pow(2).mean()
        loss.backward()
        model.means.grad[:, 2] = 0.0
        optimizer.step()
        if step == 1:
            first = loss.item()
        if step % 50 == 0 or step == 1:
            torch.mps.synchronize()
            psnr = -10 * torch.log10(loss.detach()).item()
            print(f"step {step:4d}  loss {loss.item():.5f}  psnr {psnr:.2f} dB")

    with torch.no_grad():
        final = render_hexels(model, camera)
    save_image(final.clamp(0, 1), out_dir / "fit_image_hexels_after.png")
    print(f"loss {first:.5f} -> {loss.item():.5f}; saved images to {out_dir}")


if __name__ == "__main__":
    main()

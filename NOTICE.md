# Third-party notices

The Hexels pipeline (`src/metalsplat/reference/hexels_ref.py`,
`src/metalsplat/kernels/rasterize_hexels.metal`,
`src/metalsplat/hexels_model.py`, `src/metalsplat/mcmc_hexels.py`,
`src/metalsplat/export_hexels.py`, `examples/train_hexels.py`) implements
the Hexel primitive of Kaushik, Paul and Yuille, "Honeycomb-Surfels:
Compact Hexagonal Primitives for Efficient Scene Rendering and
Reconstruction" (2026, arXiv:2610.37668). Its formulas, constants, MCMC
relocation and default hyperparameters are transcribed from the following
MIT-licensed sources:

- honeycomb-surfels/hexel-kernel, `gsplat/cuda_hexels3d/` (the original
  Hexel additions to that package):

      Copyright (c) 2026 Prakhar Kaushik, Soumava Paul, Alan Yuille

- honeycomb-surfels/hexels, `hexels/` (trainer, model, MCMC) and
  `hexel_edge/` (Metal reference):

      Copyright (c) 2026 Prakhar Kaushik and Soumava Paul
      Copyright (c) 2026 Prakhar Kaushik (Hexel Edge)

Both are distributed under the MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

No code from hexel-kernel's gsplat/Image-GS-derived (Apache-2.0) components
is included: tile binning, sorting and the rasterizer structure are
metalsplat's own. The pretrained checkpoints that
`metalsplat.export_hexels.load_hexels_checkpoint` reads are distributed
separately under their own terms (huggingface.co/jhu-ccvl/Hexels) and are
not included here.

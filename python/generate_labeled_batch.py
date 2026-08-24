"""Python/JAX port of generate_labeled_batch.m.

Generates a batch of p-bit DBM samples conditioned only on a digit label
(0-9) -- no specific MNIST example is used, since the sampler never clamps
to the visible pixels, only to the label ("sticker") bits (see
Image_generation.m).

    generate_labeled_batch(label, seed, batch_size, start_idx, output_dir)
    generate_labeled_batch(..., num_sweeps=...)  # optional, for smoke-testing

    label      : digit class to condition on (0-9)
    seed       : RNG seed for this task. MUST be unique per task -- see the
                 MATLAB docstring in generate_labeled_batch.m for why.
    batch_size : number of images to generate in this call
    start_idx  : 0-based index of the first image in this batch, used only
                 to name output files so chunks from different tasks don't
                 collide
    output_dir : base output directory; images are written to
                 <output_dir>/label_<label>/sample_<00000+start_idx>.png
                 (save_format="png") or
                 <output_dir>/label_<label>/images_<start_idx>_<end_idx>.npy
                 (save_format="npy")
    num_sweeps : optional override of the annealing sweep count per beta
                 value (default 10000, matching Image_generation.m). Only
                 meant for quick smoke tests of this script.
    save_format: "png" (default, one file per image) or "npy" (one array
                 file per call, shape (batch_size, 28, 28) uint8)
"""
import argparse
import os
import time

import numpy as np
import scipy.io as sio
from PIL import Image

from pbit_gibbs_sample import pbit_gibbs_sample

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETA = np.arange(0, 5 + 1e-9, 0.125)  # annealing schedule (same as Image_generation.m)


def _load_data(data_dir):
    colormap1 = np.loadtxt(os.path.join(data_dir, "colorMap_4264.csv"), delimiter=",").astype(int)
    required_colors = len(np.unique(colormap1))
    groups = [np.where(colormap1 == k)[0] for k in range(1, required_colors + 1)]

    Jout = sio.loadmat(os.path.join(data_dir, "Jout_100.mat"))["Jout"]
    hout = sio.loadmat(os.path.join(data_dir, "hout_100.mat"))["hout"].reshape(-1)

    index_visible = sio.loadmat(os.path.join(data_dir, "index_visible.mat"))["index_visible"].reshape(-1) - 1

    index_sticker = np.stack([
        sio.loadmat(os.path.join(data_dir, f"index_sticker{i}.mat"))[f"index_sticker{i}"].reshape(-1) - 1
        for i in range(1, 6)
    ])  # 5 x 10, 0-based

    return groups, Jout, hout, index_visible, index_sticker


def _build_bias(hout, index_sticker, NM, label, batch_size):
    hclamp = np.zeros(NM)
    hclamp[index_sticker.reshape(-1)] = -1000
    hclamp[index_sticker[:, label]] = 1000
    h_col = hout + hclamp
    return np.tile(h_col.reshape(-1, 1), (1, batch_size))


def generate_labeled_batch(label, seed, batch_size, start_idx, output_dir, num_sweeps=10000,
                            data_dir=None, test_randoms=None, progress=False, save_format="png"):
    if data_dir is None:
        data_dir = REPO_ROOT
    if save_format not in ("png", "npy"):
        raise ValueError(f"save_format must be 'png' or 'npy', got {save_format!r}")

    groups, Jout, hout, index_visible, index_sticker = _load_data(data_dir)
    NM = Jout.shape[0]

    J_bipolar = Jout
    H = _build_bias(hout, index_sticker, NM, label, batch_size)

    print(f"[label={label} seed={seed}] generating {batch_size} images "
          f"(start_idx={start_idx}, num_sweeps={num_sweeps})...")
    t0 = time.time()
    S = pbit_gibbs_sample(J_bipolar, H, groups, BETA, num_sweeps, seed=seed,
                           test_randoms=test_randoms, progress=progress)
    print(f"elapsed: {time.time() - t0:.2f}s")

    out_dir = os.path.join(output_dir, f"label_{label}")
    os.makedirs(out_dir, exist_ok=True)

    images = np.empty((batch_size, 28, 28), dtype=np.uint8)
    for b in range(batch_size):
        img = S[index_visible, b].reshape(28, 28, order="F")
        img01 = (img + 1) / 2  # bipolar {-1,+1} -> {0,1}
        images[b] = (img01 * 255).astype(np.uint8)

    if save_format == "npy":
        end_idx = start_idx + batch_size - 1
        fname = os.path.join(out_dir, f"images_{start_idx:05d}_{end_idx:05d}.npy")
        np.save(fname, images)
        print(f"[label={label}] wrote {batch_size} images to {fname}")
    else:
        for b in range(batch_size):
            fname = os.path.join(out_dir, f"sample_{start_idx + b:05d}.png")
            Image.fromarray(images[b], mode="L").save(fname)
        print(f"[label={label}] wrote {batch_size} images to {out_dir}")

    return S


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("start_idx", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("num_sweeps", type=int, nargs="?", default=10000)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--progress", action="store_true",
                         help="print progress after each beta value instead of only at the end")
    parser.add_argument("--save-format", choices=["png", "npy"], default="png",
                         help="png: one file per image (default). "
                              "npy: one (batch_size, 28, 28) uint8 array file per call")
    args = parser.parse_args()

    generate_labeled_batch(args.label, args.seed, args.batch_size, args.start_idx,
                            args.output_dir, args.num_sweeps, data_dir=args.data_dir,
                            progress=args.progress, save_format=args.save_format)

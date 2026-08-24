"""Cross-language equivalence test: pbit_gibbs_sample.m vs pbit_gibbs_sample.py.

MATLAB's Mersenne Twister and JAX's PRNG produce different, non-interchangeable
random streams even from the same integer seed, so this test does not rely on
either sampler's own RNG. Instead it generates one shared sequence of "random"
draws in Python/numpy and injects it into both implementations via the
test_randoms hook (see pbit_gibbs_sample.m and pbit_gibbs_sample.py). With
identical inputs at every step, the two implementations must produce bit
-identical output if the port is correct -- this isolates real porting bugs
(index base, matrix orientation, the tanh/sign update rule, the label clamp,
image reshape order) from RNG differences.

Run with: python python/test_equivalence.py
"""

import os
import subprocess
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_labeled_batch import BETA, _build_bias, _load_data
from pbit_gibbs_sample import pbit_gibbs_sample

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATLAB_BIN = "/Applications/MATLAB_R2026a.app/bin/matlab"

LABEL = 3
BATCH_SIZE = 1
NUM_SWEEPS = 2
SEED = 12345


def main():
    groups, Jout, hout, index_visible, index_sticker = _load_data(REPO_ROOT)
    NM = Jout.shape[0]
    required_colors = len(groups)
    H = _build_bias(hout, index_sticker, NM, LABEL, BATCH_SIZE)

    rng = np.random.default_rng(SEED)
    init = 2 * rng.random((NM, BATCH_SIZE)) - 1
    draws = [
        rng.random((len(groups[c]), BATCH_SIZE))
        for _bkk in BETA
        for _sweep in range(NUM_SWEEPS)
        for c in range(required_colors)
    ]

    draws_obj = np.empty((len(draws), 1), dtype=object)
    for i, d in enumerate(draws):
        draws_obj[i, 0] = d
    test_randoms_path = os.path.join(REPO_ROOT, "python", "_test_randoms.mat")
    sio.savemat(test_randoms_path, {"init": init, "draws": draws_obj})

    print("Running Python/JAX sampler...")
    S_py = pbit_gibbs_sample(
        Jout, H, groups, BETA, NUM_SWEEPS, test_randoms={"init": init, "draws": draws}
    )
    img_py = S_py[index_visible, 0].reshape(28, 28, order="F")

    print("Running MATLAB sampler...")
    matlab_result_path = os.path.join(REPO_ROOT, "python", "_matlab_result.mat")
    if os.path.exists(matlab_result_path):
        os.remove(matlab_result_path)
    cmd = [
        MATLAB_BIN,
        "-batch",
        f"test_pbit_equivalence({LABEL}, {BATCH_SIZE}, {NUM_SWEEPS}, "
        f"'{test_randoms_path}', '{matlab_result_path}')",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    print(proc.stdout)
    if proc.returncode != 0 or not os.path.exists(matlab_result_path):
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("MATLAB run failed")

    mat = sio.loadmat(matlab_result_path)
    S_mat = mat["S"]
    img_mat = mat["img"]

    os.remove(test_randoms_path)
    os.remove(matlab_result_path)

    s_match = np.array_equal(S_py, S_mat)
    img_match = np.array_equal(img_py, img_mat)
    s_maxdiff = np.max(np.abs(S_py - S_mat))
    img_maxdiff = np.max(np.abs(img_py - img_mat))

    print(
        f"S      : shape_py={S_py.shape} shape_mat={S_mat.shape} "
        f"exact_match={s_match} max_abs_diff={s_maxdiff}"
    )
    print(
        f"img    : shape_py={img_py.shape} shape_mat={img_mat.shape} "
        f"exact_match={img_match} max_abs_diff={img_maxdiff}"
    )

    if s_match and img_match:
        print(
            "PASS: MATLAB and Python/JAX implementations are functionally equivalent."
        )
        return 0
    else:
        print("FAIL: outputs diverge.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

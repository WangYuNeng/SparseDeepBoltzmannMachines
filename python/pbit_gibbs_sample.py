"""JAX port of pbit_gibbs_sample.m.

Chromatic (graph-colored) block Gibbs sampling with simulated annealing,
batched over independent chains. See pbit_gibbs_sample.m for the reference
MATLAB implementation this mirrors.

Two departures from a literal translation, both needed for this to actually
be fast under JAX:

1. The (beta, sweep) loop is fused into a single jax.jit-compiled lax.scan
   instead of a Python for-loop, so the whole annealing schedule dispatches
   to XLA once instead of once per sweep -- a plain Python loop is dominated
   by per-step dispatch overhead. The (static, small) color loop stays a
   Python for-loop and gets unrolled into the compiled program.

2. J_bipolar*S is computed as a fixed-degree gather ("map") + weighted sum
   ("reduce") over each node's actual neighbor list, not a dense matmul.
   The coupling graph here is genuinely sparse and locally-connected (mean
   degree ~14, max 15, out of 4264 nodes) -- a dense matmul would spend
   ~284x more FLOPs multiplying by structural zeros. Padding every row to
   the graph's max degree keeps the gather perfectly rectangular
   (jit/vmap-friendly) without paying for the zeros.
"""
import time
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)  # match MATLAB double precision
import jax.numpy as jnp
import numpy as np


def _build_neighbor_lists(J, groups):
    """For each color group, pad every row's actual nonzero neighbors to a
    common width (the graph's max degree). Padding slots point at node 0
    with weight 0, so they contribute exactly 0 to the weighted sum
    regardless of S's value there (IEEE 0*finite == 0).

    Returns (nbr_idx, nbr_w), each a tuple of one (ng(c), max_degree) array
    per color: nbr_idx are neighbor node indices, nbr_w the matching J
    weights.
    """
    max_degree = int(np.max(np.count_nonzero(J, axis=1)))
    nbr_idx, nbr_w = [], []
    for g in groups:
        rows = J[g, :]
        ng = len(g)
        idx = np.zeros((ng, max_degree), dtype=np.int32)
        w = np.zeros((ng, max_degree), dtype=np.float64)
        for i in range(ng):
            nz = np.nonzero(rows[i])[0]
            idx[i, :len(nz)] = nz
            w[i, :len(nz)] = rows[i, nz]
        nbr_idx.append(jnp.asarray(idx))
        nbr_w.append(jnp.asarray(w))
    return tuple(nbr_idx), tuple(nbr_w)


@partial(jax.jit, static_argnames=("required_colors",))
def _run_schedule_production(S0, nbr_idx, nbr_w, hg, idx_list, beta_per_step, key, required_colors):
    """Production path: draws generated on the fly from a carried PRNG key,
    so memory stays O(ng(c) x B) per step instead of O(L x ng(c) x B) for
    the whole schedule -- precomputing every draw upfront is what actually
    blew up memory at larger batch sizes, not the matmul/gather itself.
    """
    def step_fn(carry, bkk):
        S, key = carry
        for c in range(required_colors):
            neighbor_vals = S[nbr_idx[c], :]  # (ng(c), max_degree, B) -- map
            x = bkk * (jnp.sum(nbr_w[c][:, :, None] * neighbor_vals, axis=1) + hg[c])  # reduce
            key, subkey = jax.random.split(key)
            r = jax.random.uniform(subkey, hg[c].shape, dtype=jnp.float64)
            S = S.at[idx_list[c], :].set(jnp.sign(jnp.tanh(x) - 2 * r + 1))
        return (S, key), None

    (S_final, key_final), _ = jax.lax.scan(step_fn, (S0, key), beta_per_step)
    return S_final, key_final


@partial(jax.jit, static_argnames=("required_colors",))
def _run_schedule_test(S0, nbr_idx, nbr_w, hg, idx_list, beta_per_step, draws_stacked, required_colors):
    """Test-only path: draws are supplied externally (see test_randoms in
    pbit_gibbs_sample) so this sampler can be checked against
    pbit_gibbs_sample.m bit-for-bit. Only used for small-scale equivalence
    checks, so materializing the full draw sequence here is fine.
    """
    def step_fn(S, xs):
        bkk, draws_this_step = xs
        for c in range(required_colors):
            neighbor_vals = S[nbr_idx[c], :]
            x = bkk * (jnp.sum(nbr_w[c][:, :, None] * neighbor_vals, axis=1) + hg[c])
            r = draws_this_step[c]
            S = S.at[idx_list[c], :].set(jnp.sign(jnp.tanh(x) - 2 * r + 1))
        return S, None

    S_final, _ = jax.lax.scan(step_fn, S0, (beta_per_step, draws_stacked))
    return S_final


def pbit_gibbs_sample(J_bipolar, h_bipolar, groups, beta, num_sweeps, seed=0, test_randoms=None, progress=False):
    """Run the annealed chromatic Gibbs sampler.

    J_bipolar : (NM, NM) array-like coupling matrix (sparse structure, dense
                or sparse storage both accepted -- converted internally to
                a padded neighbor-list representation, see _build_neighbor_lists)
    h_bipolar : (NM, B) array-like bias matrix, one column per chain
    groups    : sequence of 1-D int arrays, 0-based p-bit indices per color
                group (bits within a group share no coupling, so they can be
                updated simultaneously)
    beta      : 1-D array, annealing (inverse temperature) schedule
    num_sweeps: sweeps to run at each beta value
    seed      : int seed for the JAX PRNG used when test_randoms is None
    test_randoms : optional dict used ONLY to cross-check this sampler
                against pbit_gibbs_sample.m. Keys:
                  "init"  : (NM, B) array substituted for the initial
                            2*rand(NM,B)-1 draw
                  "draws" : list of (ng(c), B) arrays substituted, in order,
                            for each 2*rand(ng(c),B) draw inside the (beta,
                            sweep, color) loop nest (beta outer, sweep
                            middle, color inner -- matching pbit_gibbs_sample.m)
                When omitted, draws come from the JAX PRNG (production path).
    progress  : if True (production path only), run one beta value at a time
                and print progress after each -- the whole schedule is still
                one compiled program (every chunk has the same shape, so it
                compiles once and the rest are cache hits), just split into
                len(beta) checkpoints instead of a single opaque call. Long
                num_sweeps runs can take minutes to hours, so this trades a
                little chunking overhead for visibility into how far along
                the anneal is.

    Returns S: (NM, B) numpy array, final bipolar {-1,+1} state.
    """
    h_bipolar = np.asarray(h_bipolar, dtype=np.float64)
    NM, B = h_bipolar.shape
    required_colors = len(groups)
    L = len(beta) * num_sweeps  # total (beta, sweep) steps

    J_bipolar = np.asarray(J_bipolar, dtype=np.float64)
    nbr_idx, nbr_w = _build_neighbor_lists(J_bipolar, groups)
    hg = [jnp.asarray(h_bipolar[g, :]) for g in groups]
    idx_list = [jnp.asarray(g) for g in groups]

    if test_randoms is not None:
        beta_per_step = jnp.repeat(jnp.asarray(beta, dtype=jnp.float64), num_sweeps)
        S0 = jnp.asarray(np.asarray(test_randoms["init"], dtype=np.float64))
        draws = test_randoms["draws"]
        draws_stacked = tuple(
            jnp.stack([np.asarray(draws[step * required_colors + c], dtype=np.float64) for step in range(L)])
            for c in range(required_colors)
        )
        S_final = _run_schedule_test(S0, nbr_idx, nbr_w, tuple(hg), tuple(idx_list), beta_per_step,
                                      draws_stacked, required_colors)
    else:
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        S = 2 * jax.random.uniform(init_key, (NM, B), dtype=jnp.float64) - 1

        if progress:
            t0 = time.time()
            for i, bkk in enumerate(beta):
                chunk_beta = jnp.full((num_sweeps,), bkk, dtype=jnp.float64)
                S, key = _run_schedule_production(S, nbr_idx, nbr_w, tuple(hg), tuple(idx_list),
                                                    chunk_beta, key, required_colors)
                done = i + 1
                elapsed = time.time() - t0
                eta = elapsed / done * (len(beta) - done)
                print(f"[pbit_gibbs_sample] beta={bkk:.3f} ({done}/{len(beta)}) "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
            S_final = S
        else:
            beta_per_step = jnp.repeat(jnp.asarray(beta, dtype=jnp.float64), num_sweeps)
            S_final, _ = _run_schedule_production(S, nbr_idx, nbr_w, tuple(hg), tuple(idx_list), beta_per_step,
                                                    key, required_colors)

    return np.asarray(S_final)

"""One-call BOSampler entry point, so benchmarks stay one line per arm.

    import bosampler
    res = bosampler.sample(lb, ub, act, seed_vectors, steps=300)

Runs the Interpolate loop with checkpoint cost excluded from timing (a
Recon2 coverage() rebuild costs ~1.6 s against a ~7 ms step and once
inverted a benchmark conclusion) and returns everything a benchmark row
needs.  Pass oracle=LooplessOracle(...) for the non-convex case; the loop
is identical.
"""
import time

import numpy as np

from .interpolate import Interpolate

CHECKPOINTS = (1, 5, 10, 25, 50, 100, 150, 200, 250, 300)


class _Box:
    def __init__(self, lb, ub):
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)


def sample(lb, ub, act, seed, steps=300, k=4, rng_seed=0, policy="cyclic",
           endpoint="uniform", oracle=None, poolmax=10 ** 7,
           checkpoints=CHECKPOINTS, count="attempts", max_tries=150_000,
           k_probe=32):
    """Run BOSampler; return a benchmark-row dict.

    count="attempts" (default): `steps` is the number of loop iterations,
    the historical semantics -- on the small models acceptance is ~100% so
    attempts ~= samples.  count="accepted": loop until `steps` samples have
    been ACCEPTED (or max_tries attempts), for regimes like recon2-loopless
    where the oracle rejects most blends (teleport across sign-pattern
    modes) and 300 attempts deliver ~1 sample; checkpoints then key on the
    accepted count so coverage-per-sample curves stay comparable.

    Returns dict with: samples (array, new samples only), cov, seed_cov,
    traj [(step, coverage, elapsed_s)...], seconds, ms (per delivered
    sample), attempts, unbracketed, rejected, and oracle stats when an
    oracle was used.

    k_probe is forwarded to Interpolate and only matters when the oracle
    exposes the plausibility capability (see Interpolate.__init__).
    """
    if count not in ("attempts", "accepted"):
        raise ValueError(f"count must be 'attempts' or 'accepted': {count!r}")
    smp = Interpolate(_Box(lb, ub), np.asarray(act), np.asarray(seed, float),
                      k=k, seed=rng_seed, policy=policy, endpoint=endpoint,
                      poolmax=poolmax, oracle=oracle, k_probe=k_probe)
    seed_cov = smp.cov.coverage()
    ck = set(checkpoints)
    traj, t_ck = [], 0.0
    tries = 0
    t0 = time.time()
    if count == "attempts":
        for i in range(steps):
            smp.step()
            tries += 1
            if (i + 1) in ck:
                _c = time.time()
                traj.append((i + 1, smp.cov.coverage(),
                             time.time() - t0 - t_ck))
                t_ck += time.time() - _c
    else:
        while smp.n_interp < steps and tries < max_tries:
            before = smp.n_interp
            smp.step()
            tries += 1
            if smp.n_interp > before and smp.n_interp in ck:
                _c = time.time()
                traj.append((smp.n_interp, smp.cov.coverage(),
                             time.time() - t0 - t_ck))
                t_ck += time.time() - _c
            if tries % 250 == 0:
                print(f"  [BOSampler] {smp.n_interp}/{steps} accepted, "
                      f"{tries} attempts, {time.time() - t0 - t_ck:.0f}s",
                      flush=True)
    el = time.time() - t0 - t_ck
    out = dict(samples=smp.pool.view()[len(seed):].copy(),
               cov=smp.cov.coverage(), seed_cov=seed_cov, traj=traj,
               seconds=el, ms=1000 * el / steps, attempts=tries,
               unbracketed=smp.n_fail, rejected=smp.n_rejected)
    if oracle is not None:
        out["oracle"] = dict(oracle.stats)
    return out

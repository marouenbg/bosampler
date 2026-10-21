# BOSampler

**Blend-and-Optimize Sampler** — coverage sampling of constraint-based
metabolic models, convex and loopless, with one sampling loop for both.

Coverage samplers aim to span the *support* of the flux space rather than
reproduce its density: they target the largest remaining gap in each
reaction's marginal. The established method, gapsplit, solves a quadratic
program for every sample, which does not scale to genome-scale human models.

BOSampler reaches each target by **affine blending** instead. Given two
feasible vectors that straddle the target value in the chosen reaction,
their convex combination hits that value exactly and is feasible for free:
the stoichiometry is linear and the box is convex. The convex sampler
therefore makes **no solver calls at all**.

For non-convex feasible sets — the loopless flux space being a union of
polytopes, one per sign pattern — a blend can land on a pattern neither
endpoint had, so an optional **membership oracle** certifies candidates
before they are accepted. Candidates are ranked by the coverage they would
buy *before* the oracle is consulted, so each certificate call is spent on
the most valuable sample available.

## Install

```bash
pip install -e .                 # convex sampling: numpy + scipy only
pip install -e ".[loopless]"     # adds cobra and gurobipy for the oracle
```

`cobra` and `gurobipy` are imported lazily, inside the functions that need
them, so the convex path runs with no solver installed.

## Use

One call, returning a benchmark row:

```python
from bosampler import sample

out = sample(lb, ub, act, seed, steps=300, k=4)
out["samples"]   # (n, d) array of new samples
out["cov"]       # coverage reached
out["traj"]      # [(step, coverage, seconds), ...]
```

`act` is the index of active reactions (those whose FVA range is non-empty)
and `seed` is the starting pool. The seed matters: blending can never leave
the convex hull of the pool it starts from, so coverage is capped by that
hull before a single sample is drawn. `fva_seed` builds the right one — the
`2d` FVA argmin/argmax vectors, which bracket every target in every range.

```python
from bosampler import fva_seed, Interpolate, validity
```

Driving the sampler directly:

```python
Interpolate(P, act, seed)                             # convex
Interpolate(P, act, seed, oracle=LooplessOracle(...)) # loopless
```

## Modules

| | |
|---|---|
| `interpolate` | the sampler — `Interpolate`, `Coverage`, `Pool`, `fva_seed`, `validity`, `enzyme_subsets` |
| `loopless` | `DisjunctiveOracle` / `LooplessOracle`, `fast_snp`, `LoopCheck` |
| `llfva` | loopless FVA driver — capped and uncapped, warm chains, checkpointing, witness bounds; produces seeds |
| `models` | model loaders, including a Recon 2 `.mat` path that bypasses cobra's reader |
| `run` | `sample()`, the one-call entry point |

The oracle runs a three-tier ladder so the exact certificate is the last
resort: a cache keyed on the sign pattern, then a sound-but-incomplete
topological prefilter, then the exact LP. On iND750 the prefilter removes a
third of the solves; on genome scale the cache dominates.

## Status

Research code accompanying a manuscript in preparation. The API is not yet
stable and there is no test suite. No license file yet — ask before reuse.

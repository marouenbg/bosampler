# BOSampler

**Blend-and-Optimize Sampler** — coverage sampling of convex and nonconvex constraint-based
metabolic models.

Coverage samplers aim to span the *support* of the flux space by targeting the largest remaining gap in each
reaction's marginal. They require solving a quadratic program to target each gap.

BOSampler is a solverless gap sampler, that make no solver calls in the convex case and minimal calls to simple linear
programs in the nonconvex. BOSampler enables discovery of novel metabolic flux distributions by combining speed per sample of uniform samplers
with coverage of gap samplers, which enables mapping large genome-scale models.

## Install

```bash
pip install -e .                 # convex sampling: numpy + scipy only
pip install -e ".[loopless]"     # adds cobra and gurobipy for the nonconvex
```


## Use


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

The nonconvex application only models loopless flux distributions. But a general recipe is available for the general case which consists of
a certification, a prefilter, then verification.


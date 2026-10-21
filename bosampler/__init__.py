"""BOSampler -- the Blend-and-Optimize Sampler for constraint-based models.

One sampling loop serves the convex and the non-convex (disjunctive) case:
candidates are affine blends of pool vectors aimed at coverage gaps, and an
optional oracle certifies membership where the feasible set is a union of
polytopes (e.g. thermodynamically loopless flux spaces).

    convex     Interpolate(P, act, seed)                  # oracle=None
    loopless   Interpolate(P, act, seed, oracle=LooplessOracle(...))

Modules:
    interpolate   the sampler (Interpolate), Coverage, Pool
    loopless      DisjunctiveOracle / LooplessOracle, fast_snp, LoopCheck
    llfva         native loopless FVA driver (capped/uncapped, warm chains,
                  checkpointing, witness bounds) -- the seed producer
    models        model loaders (Recon2 .mat bypass etc.)
    run           sample() -- one-call entry point returning a benchmark row
"""

from .interpolate import (Coverage, Interpolate, Pool, enzyme_subsets,
                          fva_seed, validity)
from .run import sample

__all__ = ["Coverage", "Interpolate", "Pool",
           "enzyme_subsets", "fva_seed", "validity", "sample"]

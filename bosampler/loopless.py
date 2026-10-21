"""Coverage sampling on the LOOPLESS flux space: interpolation + a tiered oracle.

The convex sampler in `interpolate.py` needs no solver at all: `Sv = b` and the box
are both preserved by convex combination, so a blend of feasible points is feasible
by algebra.  The loop law is not preserved that way -- it is a condition on the SIGN
PATTERN, and a blend can land on a pattern neither endpoint had.  So a blend must be
CHECKED.  This module is that check, and the sampler wrapped around it.

NOT EVERY SAMPLE NEEDS A SOLVE.  That is the whole design.  Because the loop law
constrains only the sign pattern, feasibility is a property of the PATTERN, not of
the point: if pattern p is loop-law-feasible then every v in P with sign(v_C) = p is
loopless.  So the oracle is a ladder and only the last rung calls a solver:

  tier 0  PRE-SEEDED PATTERNS.  Every pool vector is loopless, so its pattern is
          feasible and is loaded into the cache up front.  A blend of two vectors
          that agree in sign on every cyclic reaction has EXACTLY that pattern
          (a blend of two same-signed values keeps the sign; two zeros stay zero),
          so it hits this tier free, as a theorem rather than a heuristic.
          MEASURED WORTHLESS ON iND750 and kept only because it costs nothing:
          84.9% -> 85.3% of checks still needed an LP, i.e. no effect.  With 64
          cyclic reactions the pattern space is vast and blends of 931 seed
          vectors essentially never reproduce a seed vector's exact pattern.
          Expect it to matter only where |C| is small.
  tier 1  CACHE.  Any pattern already decided, either way.  Free.  Absorbs 15% on
          iND750, but 92% on iJR904 (944 solves for 11486 samples) -- reuse is
          the dominant term on some models and absent on others.
  tier 2  PREFILTER (optional).  ll-ACHRB's topological test: v is loopy iff
          sign(v) aligns with a Fast-SNP basis cycle on its support.  ~2 us, and
          SOUND WHEN IT FIRES -- 0 over-strict verdicts in 8450 validation points
          -- so a rejection here is final and needs no LP.  It is NOT complete
          (2 misses on iND750 loopless blends), so a PASS still falls through.
          THE ONE THAT PAYS: 13353 -> 8981 solves on iND750 (84.9% -> 56.3% of
          checks), 9% more samples, coverage and validity unchanged.
  tier 3  LP ORACLE.  Fix every indicator binary to the pattern; the MILP collapses
          to an LP.  Exact.  Measured 0.4-8.1 ms/call.

Consolidated here from eleven copies across `nonconvex/` (`loopless_4arm.py`,
`loopless_ind750_fair.py`, `loopless_headtohead.py`, ...).  Two bugs already
traced to that duplication: a hardcoded `'100.0%'` validity string that was never
measured, and an oracle cache shared across arms that handed later arms the
patterns earlier ones had solved.

DO NOT SEED FROM UNVERIFIED VECTORS.  `cobra`'s cycleFreeFlux route
(`loopless_fva_iter`) is ~2x faster than the exact MILP and gives a better reach
ceiling, but 4% of its vectors carry loops -- its second branch removes only "loops
including the current reaction" and leaves others.  A loopy seed vector poisons
every blend drawn from it, and tier 0 would then certify blends against a pattern
that is not actually feasible.  `filter_loopless` exists for exactly this.
"""

import collections
import time

import numpy as np

from .interpolate import Interpolate


# ---------------------------------------------------------------------------
# Fast-SNP and the topological loop test.
#
# Ported from github.com/SysBioengLab/looplessFluxSampler (MIT, Pedro A. Saa),
# `fxns/fast_snp.m` and `fxns/looplessCheck.m`, and used here ONLY as tier 2 --
# a sound-but-incomplete prefilter in front of the exact LP.  Measured on
# iND750: 13353 -> 8981 solves (33% fewer), 9% more samples, validity still
# 100%.  Trusting a PASS instead of falling through to the LP drops validity to
# 95%; see the module docstring.
# ---------------------------------------------------------------------------

EPS_SNP = 1e-1          # fast_snp.m: epsilon
VTOL = 1e-6             # fast_snp.m: vTol
BND = 1e2               # looplessFluxSampler.m:125 -- +-100 on reversible internals


# ---------------------------------------------------------------------------
# fast_snp.m -- Sparse Null-space Pursuit
# ---------------------------------------------------------------------------
def fast_snp(S, lb, ub, vtol=VTOL, max_cols=None, verbose=False):
    """Sparse basis of the DIRECTIONALLY FEASIBLE null space of S.

    min ||v||_1  s.t.  Sv = 0,  x'v >= eps,  lb <= v <= ub, solved repeatedly with
    x drawn at random and projected orthogonal to the columns already found.  The
    bounds are what keep the basis small: an irreversible reaction gets lb = 0, so
    cycles that would need it reversed are never generated.
    """
    import gurobipy as gp

    n = S.shape[1]
    N = []
    rng = np.random.default_rng(0)
    env = gp.Env(params={"OutputFlag": 0})

    # L1 via v = vp - vn.  Sv = 0 and the box act on v; the objective on vp + vn.
    md = gp.Model(env=env)
    vp = md.addMVar(n, lb=0.0, ub=np.maximum(ub, 0.0))
    vn = md.addMVar(n, lb=0.0, ub=np.maximum(-lb, 0.0))
    v = vp - vn
    md.addConstr(S @ v == 0)
    md.setObjective(vp.sum() + vn.sum(), gp.GRB.MINIMIZE)
    sense = md.addConstr(vp.sum() - vn.sum() >= EPS_SNP)   # placeholder, respecified

    nullity = n - np.linalg.matrix_rank(S) if n < 4000 else n
    cap = nullity if max_cols is None else min(nullity, max_cols)
    t0 = time.time()
    for _ in range(cap):
        # x = (1 + weights) * (I - P_N' P_N):  a random row orthogonal to span(N)
        w = 1.0 + rng.random(n)
        if N:
            Q, _ = np.linalg.qr(np.array(N).T)      # orth(Nsnp)
            w = w - (w @ Q) @ Q.T
        if np.linalg.norm(w) < 1e-12:
            break

        best = None
        for sgn in (1.0, -1.0):
            md.remove(sense)
            sense = md.addConstr(sgn * (w @ vp - w @ vn) >= EPS_SNP)
            md.optimize()
            if md.Status != gp.GRB.OPTIMAL:
                continue
            cand = vp.X - vn.X
            cand[np.abs(cand) < vtol] = 0.0
            k = int(np.count_nonzero(cand))
            if k and (best is None or k < best[0]):
                best = (k, cand)
        if best is None:
            break                                    # no feasible direction left
        N.append(best[1])
        if verbose and len(N) % 25 == 0:
            print(f"    fast_snp: {len(N)} cols, {time.time()-t0:.0f}s", flush=True)

    md.dispose()
    env.dispose()
    return np.array(N).T if N else np.zeros((n, 0))


# ---------------------------------------------------------------------------
# looplessCheck.m -- the topological test
# ---------------------------------------------------------------------------
class LoopCheck:
    """`sample.isFeasible`: v is loopless iff no basis cycle is sign-aligned."""

    def __init__(self, Nint, int_rxns):
        self.int_rxns = np.asarray(int_rxns, int)
        self.L = np.sign(Nint)                       # loopMatrix = sparse(sign(Nint))
        self.nnz = np.count_nonzero(Nint, axis=0)    # nnzEntries = sum(Nint ~= 0)

    def __call__(self, V):
        """V is (k, n) or (n,).  Returns a boolean vector: True = loopless."""
        one = V.ndim == 1
        V = np.atleast_2d(V)
        if self.L.shape[1] == 0:
            return np.ones(len(V), bool)[0] if one else np.ones(len(V), bool)
        sg = np.sign(V[:, self.int_rxns])            # (k, ni)
        score = np.abs(sg @ self.L)                  # (k, p)
        loopy = (score == self.nnz[None, :]).any(axis=1)
        return (~loopy)[0] if one else ~loopy



class DisjunctiveOracle:
    """Membership oracle for MILP-representable (disjunctive) feasible sets.

    THE GENERAL PROBLEM.  Sample  X = { v in P : Phi_sigma(v) feasible }  where
    P is the base polytope (Sv = b and the box -- handled for free by blending),
    `sigma` maps a point to a discrete LABEL (the sign pattern of affine switch
    functionals; tiles = its level sets), and `Phi_p` is a per-label certificate
    problem in auxiliary variables only.  By Jeroslow-Lowe this is exactly the
    MILP-representable class: every finite union of polyhedra, and nothing more.

    The contract:

      signature(v) -> np.int8 array   arithmetic; a 0 entry is a wildcard switch
      certify(label) -> bool          EXACT -- correctness lives here
      prefilter(v) -> bool, optional  SOUND on False: a reject is final and is
                                      cached; a pass falls through to certify

    Everything else is problem-independent: the label-keyed cache (one solve
    settles an entire tile forever -- the reason the oracle is cheap at all),
    the tier ladder, the stats.  Correctness needs only `certify` exact; the
    disjunctive structure is what makes membership CACHEABLE rather than a
    fresh solve per point.

    Instantiations: loopless (sigma = sign of cyclic fluxes, Phi = potential LP;
    see LooplessOracle below), thermodynamic/TFA (same sigma, same LP with the
    potentials boxed to mu'0 +- RT ln of the concentration window), Boolean TRN
    rules (Phi = rule consistency), capped gene budgets (Phi = arithmetic).
    """

    def __init__(self, signature, certify, prefilter=None, seed_vectors=None):
        self.signature = signature
        self.certify = certify
        self.prefilter = prefilter
        self.cache = {}
        self.stats = dict(seeded=0, cache=0, prefilter=0, lp=0, accepted=0,
                          rejected=0)
        if seed_vectors is not None:
            self.seed_labels(seed_vectors)

    def seed_labels(self, vectors):
        """Prime known-feasible labels (e.g. the seed's own tiles).  Trusted
        forever after -- only call with vectors verified to be in X."""
        for v in np.atleast_2d(np.asarray(vectors, float)):
            self.cache.setdefault(self.signature(v).tobytes(), True)

    def __call__(self, v):
        key = self.signature(v).tobytes()
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache"] += 1
            self.stats["accepted" if hit else "rejected"] += 1
            return hit
        if self.prefilter is not None and not bool(self.prefilter(v)):
            self.cache[key] = False
            self.stats["prefilter"] += 1
            self.stats["rejected"] += 1
            return False
        ok = bool(self.certify(self.signature(v)))
        self.cache[key] = ok
        self.stats["lp"] += 1
        self.stats["accepted" if ok else "rejected"] += 1
        return ok

    @property
    def solves(self):
        return self.stats["lp"]

    def report(self):
        s = self.stats
        tot = s["cache"] + s["prefilter"] + s["lp"]
        return (f"{tot} checks: {s['cache']} cached, {s['prefilter']} "
                f"prefiltered, {s['lp']} solved "
                f"({100*s['lp']/max(tot,1):.1f}% needed a certificate); "
                f"{s['accepted']} accepted, {s['rejected']} rejected")


class LooplessOracle(DisjunctiveOracle):
    """Tiered loopless feasibility test.  Call it with a flux vector.

    `model` must ALREADY have `add_loopless` applied and its objective set to a
    constant -- this asks feasibility, not optimality, and leaving a real
    objective in place silently turns every call into an optimisation.  The caller
    owns the model context; this class only flips indicator bounds and restores
    them.

    `prefilter` is an optional callable v -> bool (True = looks loopless), used as
    tier 2.  It must be SOUND: a False verdict is taken as final and no LP is run.
    `nonconvex/ll_achrb.LoopCheck` satisfies this.
    """

    def __init__(self, model, cyclic_ids, reaction_ids, prefilter=None,
                 seed_vectors=None, ztol=1e-9, harvest=False,
                 rho0=0.95, learn_window=256, rej_window=64, ptol=1e-6):
        self.model = model
        self.cyc = list(cyclic_ids)
        self.ztol = ztol
        rid = list(reaction_ids)
        self.cyc_j = np.array([rid.index(c) for c in self.cyc], dtype=int)
        self.ind = {c: model.solver.variables[f"indicator_{c}"] for c in self.cyc}
        # --- plausibility capability state (see `plausibility` below) ---
        # rho0: prior per-risk-bit survival probability, refit online.
        # learn_window: sliding window of LP verdicts the fit sees.
        # rej_window: how many LP-rejected patterns the zero rule keeps.
        # ptol: RANKING deadband for the bitsets (never affects certification).
        self.ptol = float(ptol)
        self.rho = float(rho0)
        self._pat_rows = {}                  # pool row index -> int8 cyclic pattern
        self._pending = {}                   # predicted union key -> novel bits
        self._verdicts = collections.deque(maxlen=int(learn_window))
        self._rej_supports = collections.deque(maxlen=int(rej_window))
        self._n_fit = 0
        self.nb_checked = []                 # novel bits of oracle-checked cands
        self.nb_lp = []                      # ... the subset that reached the LP
        self._pending_miss = 0               # checked pattern != predicted union
        # HARVEST: a feasible certificate LP does not just answer yes -- its
        # solution IS a loopless flux vector, sitting in the newly certified
        # tile and generally OUTSIDE conv(pool).  Since accepted blends can
        # never grow the hull (conv invariant), these free by-products are the
        # only hull-growing loopless vectors the sampler ever sees.  Drain
        # `harvested` into the pool (NOT the coverage record: they are not
        # samples) to bootstrap reach from a small verified seed.
        self._harvest = bool(harvest)
        self.harvested = []
        # The loopless case is one instantiation of the general contract:
        #   signature = sign pattern on cyclic reactions (affine switches v_i)
        #   certify   = fix the indicators to the label, ask the LP for potentials
        super().__init__(signature=self.pattern, certify=self._solve,
                         prefilter=prefilter, seed_vectors=seed_vectors)
        self.stats["rho"] = self.rho
        self.stats["plaus_scored"] = 0
        self.stats["plaus_zeroed"] = 0

    def pattern(self, v):
        """Sign pattern on cyclic reactions, with a deadband around zero.

        The deadband matters in BOTH directions and 1e-9 is not arbitrary: a
        residual flux counted as active can complete a spurious cycle, while one
        counted as zero leaves its indicator free and can pass a pattern that is
        not really attainable.  Measured on iND750: no cyclic coordinate of a
        CycleFreeFlux vector falls between 1e-9 and 1e-7, so verdicts there are
        flat across that whole range and the choice is not load-bearing.
        """
        c = v[self.cyc_j]
        return np.sign(np.where(np.abs(c) > self.ztol, c, 0.0)).astype(np.int8)

    def seed_patterns(self, vectors):
        """Historical name for `seed_labels` -- known-loopless tiles, trusted."""
        self.seed_labels(vectors)

    # ------------------------------------------------------------------
    # Plausibility capability -- consumed generically by Interpolate.step,
    # which only ever sees a float in [0, 1] per candidate pair.  ALL
    # loop-law reasoning about candidate RANKING lives here.
    #
    # THE GEOMETRY (measured on the Recon2 llfva seed at ptol=1e-6).  For a
    # convex blend v = lam*A + (1-lam)*B, coordinate-wise:
    #   * both parents zero        -> blend zero
    #   * one active, one zero     -> blend ACTIVE with the active parent's
    #                                 sign (any interior lam)
    #   * both active, same sign   -> blend active, that sign
    #   * both active, OPPOSITE    -> blend active at generic lam, but its
    #                                 SIGN depends on lam ("conflict")
    # So a blend's cyclic SUPPORT is the union of its parents' supports
    # (modulo the deadband), computable before blending.  The sign pattern
    # is the union too, EXCEPT on conflicted coordinates -- and conflicts
    # are real on this seed: 1265 of 3842 cyclic reactions take both signs
    # at full flux magnitude across seed members (a random pair carries a
    # median 207 novel support bits and 23 conflicted signs).
    #
    # THE MODEL.  Feasibility falls sharply with how far the blend's tile
    # sits from a certified one, measured in
    #
    #     risk = novel_bits + conflict_bits
    #     novel_bits    = |union support| - max(|supp A|, |supp B|)
    #     conflict_bits = #{i : sign A_i * sign B_i < 0}
    #
    # (each novel bit fixes one more indicator the larger parent's
    # certificate never had to satisfy; each conflict is a sign the blend
    # may FLIP relative to that certificate), and the prior is
    #
    #     P(pass | pair) = rho ** risk,
    #
    # rho learned online from this oracle's own LP verdicts (sliding-window
    # 1-D MLE, `_refit_rho`).  Floored at 1e-250: when every candidate
    # underflows, ranking degrades to plain gain order, never to formation
    # order.  Refinements:
    #
    #   * risk == 0: the blend's ptol-pattern IS the larger parent's -- a
    #     pool pattern, hence certified.  rho**0 == 1.  (Near-sound rather
    #     than a theorem only because sub-ptol activations can differ.)
    #   * the union is a sign-consistent superset of an LP-REJECTED
    #     pattern, with no conflict touching that pattern's support: the
    #     blend then reproduces the rejected fixings and adds more, so the
    #     certificate LP only shrinks -- scored 0.  (Again exact modulo
    #     sub-ptol bits.)
    #
    # Neither shortcut certifies anything downstream: the sampler uses
    # these numbers for RANKING only, and every accepted sample still
    # passes the full oracle above.
    #
    # TOLERANCE.  Ranking bitsets use ptol (default 1e-6, the benchmark's
    # own activity tolerance), NOT the oracle's ztol=1e-9: at 1e-9 the
    # bitsets would count solver noise as structure.  ptol touches ranking
    # and learning only -- certification stays at ztol.
    #
    # DESIGN CHOICE (bitset provenance): per-pool-row patterns are built
    # LAZILY from the rows the sampler actually pairs, read straight out of
    # `pool.buf`, and cached by ROW INDEX (`_pat_rows`).  No constructor
    # argument, no up-front pass over the seed.  The cache assumes the pool
    # is append-only -- true under bosampler.sample's poolmax=1e7 default;
    # if a bounded pool ever compacts, stale entries degrade ranking
    # quality only, never correctness.
    # ------------------------------------------------------------------

    def _rank_pattern(self, v):
        """Sign pattern at the RANKING tolerance ptol (not the oracle's ztol)."""
        c = v[self.cyc_j]
        return np.sign(np.where(np.abs(c) > self.ptol, c, 0.0)).astype(np.int8)

    def _row_patterns(self, rows, pool):
        miss = [r for r in dict.fromkeys(int(r) for r in rows)
                if r not in self._pat_rows]
        if miss:
            block = np.asarray(pool.buf[np.asarray(miss, int)[:, None],
                                        self.cyc_j[None, :]])
            sg = np.sign(np.where(np.abs(block) > self.ptol, block, 0.0))
            for r, p in zip(miss, sg.astype(np.int8)):
                self._pat_rows[r] = p
        return np.stack([self._pat_rows[int(r)] for r in rows])

    def plausibility(self, ia, ib, pool):
        """Prior in [0, 1] that the blend of pool rows ia[t], ib[t] passes."""
        SA = self._row_patterns(np.asarray(ia, int), pool)
        SB = self._row_patterns(np.asarray(ib, int), pool)
        U = np.where(SA != 0, SA, SB)          # sign union; ambiguous on conflicts
        supp = (SA != 0) | (SB != 0)
        novel = supp.sum(1) - np.maximum((SA != 0).sum(1), (SB != 0).sum(1))
        conf = (SA.astype(np.int16) * SB.astype(np.int16) < 0)
        risk = novel + conf.sum(1)
        w = np.maximum(self.rho ** risk.astype(float), 1e-250)
        if self._rej_supports:
            R = np.stack(list(self._rej_supports))               # (r, C) int8
            Rs = R != 0
            covered = ~np.any(Rs[None, :, :] & ~supp[:, None, :], axis=2)
            clean = ~np.any(conf[:, None, :] & Rs[None, :, :], axis=2)
            signed = ~np.any((U[:, None, :] != R[None, :, :]) & Rs[None, :, :],
                             axis=2)
            dead = (covered & clean & signed).any(1)
            if dead.any():
                w[dead] = 0.0
                self.stats["plaus_zeroed"] += int(dead.sum())
        # remember what was predicted (keyed by the blend's expected SUPPORT,
        # which unlike the sign union is conflict-proof), so __call__ can turn
        # its verdicts into (risk, outcome) training points for the rho fit
        self._pending = {supp[t].tobytes(): int(risk[t])
                         for t in range(len(supp))}
        self.stats["plaus_scored"] += len(supp)
        return w

    def __call__(self, v):
        if not self._pending:                  # capability not in use: as before
            return super().__call__(v)
        rp = self._rank_pattern(v)
        nb = self._pending.get((rp != 0).tobytes())
        lp0 = self.stats["lp"]
        ok = super().__call__(v)
        if nb is None:
            # blend support differs from the predicted union (deadband
            # shrinkage or exact cancellation) -- no learning signal here
            self._pending_miss += 1
            return ok
        self.nb_checked.append(nb)
        if self.stats["lp"] > lp0:             # a fresh certificate verdict
            self.nb_lp.append(nb)
            if not ok:
                self._rej_supports.append(rp)
            if nb >= 1:                        # rho**0 == 1 carries no signal
                self._verdicts.append((nb, bool(ok)))
                self._n_fit += 1
                if self._n_fit % 16 == 0:
                    self._refit_rho()
        return ok

    def _refit_rho(self):
        """Sliding-window MLE of rho in P(pass | risk n) = rho**n.

        1-D likelihood on a fixed grid, geometrically dense near 1 where
        rho**n at n of a few hundred actually discriminates: simple,
        bounded away from 0 and 1, no iterative failure mode.
        """
        if not self._verdicts:
            return
        nb = np.array([t[0] for t in self._verdicts], float)
        y = np.array([t[1] for t in self._verdicts], bool)
        grid = 1.0 - np.geomspace(0.002, 0.95, 240)
        with np.errstate(under="ignore"):
            p = np.clip(grid[None, :] ** nb[:, None], 1e-300, 1.0)
            ll = np.where(y[:, None], np.log(p),
                          np.log(np.clip(1.0 - p, 1e-12, None))).sum(0)
        self.rho = float(grid[int(np.argmax(ll))])
        self.stats["rho"] = self.rho

    def _solve(self, sig):
        for c, sv in zip(self.cyc, sig):
            u = self.ind[c]
            u.lb, u.ub = (1, 1) if sv > 0 else ((0, 0) if sv < 0 else (0, 1))
        try:
            self.model.solver.optimize()
            ok = self.model.solver.status == "optimal"
            if ok and self._harvest:
                pv = self.model.solver.primal_values
                self.harvested.append(np.array(
                    [pv[r.forward_variable.name] - pv[r.reverse_variable.name]
                     for r in self.model.reactions]))
        except Exception:
            ok = False
        for c in self.cyc:                       # ALWAYS restore, even on throw
            self.ind[c].lb, self.ind[c].ub = 0, 1
        return ok


def cyclic_reactions(loopless_model):
    """Reaction ids carrying a loop-law indicator, in a model after add_loopless."""
    return sorted({v.name[len("indicator_"):]
                   for v in loopless_model.solver.variables
                   if v.name.startswith("indicator_")})


def filter_loopless(vectors, oracle, verbose=False):
    """Drop vectors that are not loopless.  Use before seeding from ANY source
    that is not the exact MILP.

    cobra's cycleFreeFlux route leaves loops in ~4% of vectors on iND750 (its
    second branch removes only loops through the target reaction), and an unclean
    seed is worse than a slow one: tier 0 would certify blends against a pattern
    that is not actually feasible, so the damage is silent and unbounded.
    """
    V = np.atleast_2d(np.asarray(vectors, float))
    keep = np.array([bool(oracle(v)) for v in V])
    if verbose:
        print(f"  filter_loopless: {int(keep.sum())}/{len(V)} kept "
              f"({100*(1-keep.mean()):.1f}% carried loops)", flush=True)
    return V[keep]

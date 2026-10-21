"""Solver-free gapsplit by convex interpolation -- the working convex sampler.

gapsplit pays one QP per sample to land a chosen reaction on a chosen value.
That solve is unnecessary whenever two points already in hand STRADDLE the target
in that coordinate: if A_j > tau > B_j then

    lam = (tau - B_j) / (A_j - B_j)   in (0, 1)
    v   = lam*A + (1 - lam)*B

hits v_j = tau exactly, satisfies Sv = b exactly (both endpoints do, and the
constraint is linear), and stays inside the box automatically (a convex
combination of two boxed points is boxed).  No solver, no tolerance, no failure
mode -- just arithmetic.

MEASURED at the tuned settings (full FVA seed, k=16, endpoint="spread"), equal
wall clock of 120 s per arm, both arms seeded identically, gapsplit at ITS OWN
best solver configuration per model.  Mean gap = 1 - coverage, lower is better:

    model         reactions   gapsplit   interpolation   improvement
    e_coli_core          87     0.0048          0.0023          52%
    iJR904              666     0.0419          0.0087          79%
    Recon2             7324     0.1430          0.0392          73%

Getting the BASELINE right moved these more than any change to our side.  Gating
gapsplit on Status==OPTIMAL discarded its SUBOPTIMAL solutions -- perfectly
feasible, just not certified -- which on iJR904 alone cost it a factor of 3.5
(0.1489 -> 0.0419).  And no single solver setting is best everywhere:
NumericFocus=2 is worth having on iJR904, where 2059 of 14070 QPs return nothing
at all, and costs 2.4x per step on Recon2 where every solve is already OPTIMAL.
Report gapsplit at its own best config or the comparison is not worth reading.

ACCURACY, which is robust to all of that.  On iJR904 gapsplit's median |Sv-b| is
8.6e-06 against interpolation's 9.0e-13.  A convex combination of two vectors with
residual r has residual <= r, so interpolation INHERITS its seed's accuracy while
a solver resets to its own tolerance on every sample.  Do not compare the two with
a pass/fail validity threshold: any cut between 1e-12 and 1e-5 turns a smooth
seven-order-of-magnitude difference into a meaningless 0%/100%.

WHERE IT LOSES.  e_coli_core, on wall clock: gapsplit is FASTER per step there
(0.9 ms vs 1.3) because the QP is trivial and k=16 candidate scoring dominates.
The tuning below is derived on Recon2 and is wrong for small models -- drop k.

WHEN THIS IS VALID.  Only on a CONVEX feasible set, or on a non-convex one whose
discrete structure admits a JOIN.  With assignments y_A and y_B behind the two
endpoints, the blend needs every reaction active that either endpoint needed, so
it is certified for free exactly when y_A OR y_B is itself legal:

    GPR (and/or only)   monotone           -> join exists   100% certified
    gene budget         monotone but capped-> join may bust    0% once tight
    TRN (`not` in 40%)  not monotone       -> no join        37% certified
    loopless            sign patterns      -> no join        fails

So `interpolate` is for convex models.  For the others the segment A->B still
carries information -- the feasible subset of lam is a union of intervals whose
complement maps the holes -- but that needs a feasibility oracle, and lives in
the non-convex module rather than here.

USE EVERY FVA SOLUTION VECTOR.  This is the only knob that matters, and getting
it wrong once cost 43% of the achievable coverage.  Every sample lies in
conv(pool), so the seed's hull is a hard ceiling on reach.  Swept on Recon2, 300
steps, K=8 (mean gap, lower is better; `unbrack` = steps where no pool pair
straddled the target):

    seed  1200 -> mean gap 0.2016, 43 unbracketed
    seed  2400 ->          0.1761, 27
    seed  4800 ->          0.1475, 14
    seed  9600 ->          0.1267,  4
    seed 14648 ->          0.1158,  0      <- all of them, and free

Everything else is third-order by comparison: K=2->32 buys 5.9% and saturates at
16; `endpoint="spread"` buys 0.9% at K=8 and 2.4% at K=32 but costs 28% more time
there.

THOSE ARE EQUAL-STEP NUMBERS AND `endpoint` REVERSES AT EQUAL WALL CLOCK.  An
earlier version of this paragraph recommended `endpoint="spread"` two lines after
reporting that it costs 28% more time -- reading a per-step gain as if it were
free.  A knob that helps per step and costs time per step is a loss unless step
count is what you are billed for.

`endpoint="uniform"` is confirmed twice and independently: by the sweep above, and
by `examples/autotune_bench.py`, which optimises against the clock on Recon2 and
converged to

    {"k": 64, "wspread": 0.0011, "policy": "worstgap", "endpoint": "uniform"}

That run also settles `wspread`: it went to ~0, so the gap-vs-spread mix is not
worth carrying.  It does NOT settle `k`.  The sweep says K=2->32 buys 5.9% and
saturates at 16; the autotuner wanted 64 on Recon2.  Bigger k costs K flops per
step and buys better incidental coverage, so the optimum moves with model size and
with how expensive a step already is -- k=8 here is a middle default, not a
measured winner.  Tune it per model if it matters; do not quote a single value.

The one first-order knob is THE FULL FVA SEED.  Everything else is trim.

The seed is free -- FVA produces 2 solution vectors per reaction as a by-product
of the bounds you needed anyway.  Sampling it down is pure loss.

TWO RECORDS, NEVER ONE.  gapsplit writes the bound vectors into samples[0] and
samples[1] to anchor its metric.  Those anchors are NOT feasible -- they satisfy
the box but violate Sv = b -- and being extremal in every coordinate they sit at
one end of the largest gap almost always.  Putting them in the interpolation pool
took validity to 1.7%, and it looks exactly like a geometric result rather than a
bug.  `Coverage.raw` holds the anchors; `Interpolate.pool` never does.
"""

import time

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


class Pool:
    """Append-only vector pool with O(1) column access.

    The obvious implementation -- keep a Python list and call `np.asarray(pool)`
    each step -- copies the WHOLE pool to read ONE column.  At 4000 vectors x
    1075 reactions that is 4.3M floats (34 MB) per step, and it dominated
    everything: profiled at 4.1 ms/sample early in a run and 174.3 ms/sample
    later, against ~2 ms for the actual feasibility oracle.  It made the sampler
    look like it saturated with budget when it was only saturating on memcpy.

    Backing the pool with a preallocated array and slicing `buf[:n, j]` makes a
    column read O(n) instead of O(n*d), with amortised O(1) appends by capacity
    doubling.  `maxlen` keeps the most recent vectors when the cap is hit.
    """

    def __init__(self, vecs, maxlen=None):
        v = np.atleast_2d(np.asarray(vecs, float))
        self.n, self.d = v.shape
        # THE SEED IS NEVER EVICTED.  conv(pool) = conv(seed) is invariant, so the
        # seed vectors ARE the reachable set -- dropping them shrinks the hull
        # mid-run and permanently lowers the coverage ceiling.  They are also the
        # oldest rows, which is exactly what a naive ring buffer discards first.
        self.n_seed = self.n
        self.maxlen = maxlen
        cap = max(1024, 2 * self.n)
        if maxlen:
            # maxlen bounds GROWTH past the seed, not the seed itself.  Clamping
            # cap below n instead raised `could not broadcast (14648,d) into
            # (8002,d)` -- the default poolmax=4000 could not hold a genome-scale
            # FVA seed at all, so every caller had to pass poolmax=10**7.
            #
            # NEVER preallocate maxlen rows: with poolmax=10**7 that asked for
            # 444 GiB up front (10**7 x 5,956 x 8 B) and was refused on 192 GB
            # cluster nodes (Linux overcommit heuristic), killing every subtype
            # sampling job -- it only ever "worked" where the OS handed out
            # untouched virtual pages.  Capacity doubles on demand in `append`.
            cap = min(max(cap, self.n + 1), 2 * maxlen + 2)
            cap = max(cap, self.n + 1)
        self.buf = np.empty((cap, self.d))
        self.buf[:self.n] = v

    def __len__(self):
        return self.n

    def col(self, j):
        return self.buf[:self.n, j]

    def __getitem__(self, i):
        return self.buf[i]

    def view(self):
        return self.buf[:self.n]

    def append(self, v):
        if self.maxlen and self.n >= max(self.maxlen, self.n_seed + 2):
            # Evict the oldest INTERPOLATED half, keeping every seed vector.
            # Evicting seed rows would shrink conv(pool) and cap coverage lower
            # than it started -- the one thing the pool must never do.
            n_interp = self.n - self.n_seed
            keep = n_interp // 2
            self.buf[self.n_seed:self.n_seed + keep] = self.buf[self.n - keep:self.n]
            self.n = self.n_seed + keep
        elif self.n == len(self.buf):
            nb = np.empty((2 * len(self.buf), self.d))
            nb[:self.n] = self.buf[:self.n]
            self.buf = nb
        self.buf[self.n] = v
        self.n += 1


class Coverage:
    """gapsplit's max-gap state, with row-wise binary-search scoring.

    Scoring every reaction against a candidate is the inner loop, and a linear
    scan over the sorted sample matrix costs O(n_active * m) per candidate.  A
    vectorised binary search does the same work in ceil(log2(m+1)) passes -- at
    m = 7324 that is 13 passes instead of 7324 comparisons per row.
    """

    def __init__(self, idx, lo, hi, refresh=50):
        self.idx = np.asarray(idx)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.rng = np.maximum(self.hi - self.lo, 1e-12)
        self.nA = len(self.idx)
        self.rows = np.arange(self.nA)
        self.refresh = refresh
        self.raw = [self.lo.copy(), self.hi.copy()]      # ANCHORS, not samples
        self._last = 0
        self._rebuild()

    def _rebuild(self):
        self._last = len(self.raw)
        self.SV = np.sort(np.array(self.raw).T, axis=1)
        d = np.diff(self.SV, axis=1)
        self.mg = d.max(1)
        self.mg2 = (np.partition(d, -2, axis=1)[:, -2] if d.shape[1] > 1
                    else np.zeros(self.nA))
        self.m = self.SV.shape[1]
        self.nbis = max(1, int(np.ceil(np.log2(self.m + 1))))

    def _locate(self, x):
        lo = np.zeros(self.nA, np.int64)
        hi = np.full(self.nA, self.m, np.int64)
        for _ in range(self.nbis):
            mid = (lo + hi) >> 1
            go = self.SV[self.rows, np.minimum(mid, self.m - 1)] < x
            lo = np.where(go, mid + 1, lo)
            hi = np.where(go, hi, mid)
        return lo

    def add(self, vec, force=False):
        self.raw.append(np.asarray(vec)[self.idx])
        # ADAPTIVE refresh.  A rebuild is O(m log m) over the whole record, so
        # doing it every fixed number of adds costs O(m log m / c) per add and
        # GROWS with m.  With the full 14648-vector FVA seed on Recon2 the record
        # starts at ~107M entries and a fixed refresh=50 would spend more time
        # re-sorting than sampling.  Rebuilding every m/20 adds keeps the
        # amortised cost bounded instead; the price is slightly staler targets
        # between rebuilds, which matters less the larger the record is.
        since = len(self.raw) - self._last
        if force or since >= max(self.refresh, self._last // 20):
            self._rebuild()

    def gain_of(self, vec):
        """Change in coverage this candidate would buy, over all reactions.

        A candidate only helps a reaction whose CURRENT largest gap it splits;
        after the split that reaction's largest gap becomes the bigger half, or
        the previous runner-up gap if that is larger -- hence mg2.
        """
        x = np.asarray(vec)[self.idx]
        k = self._locate(x)
        ok = (k > 0) & (k < self.m)
        if not ok.any():
            return 0.0
        r = self.rows[ok]
        kk = k[ok]
        left = self.SV[r, kk - 1]
        right = self.SV[r, kk]
        inmax = (right - left) >= self.mg[r] - 1e-15
        new = np.maximum(x[ok] - left, right - x[ok])
        nm = np.maximum(new, self.mg2[r])
        return float(np.where(inmax, (self.mg[r] - nm) / self.rng[r], 0.0).sum()
                     / self.nA)

    def gap_gains(self):
        """Best achievable gain per reaction, if its largest gap were halved."""
        g = self.mg
        return (g - np.maximum(self.mg2, g / 2.0)) / self.rng / self.nA

    def target_of(self, a):
        row = self.SV[a]
        d = np.diff(row)
        k = int(np.argmax(d))
        return 0.5 * (row[k] + row[k + 1])

    def coverage(self, exact=True):
        """Current coverage.  `exact=True` REBUILDS, and that is not cheap.

        A rebuild re-sorts the whole record: 1582 ms on Recon2 with the full
        14648-vector seed, against 3.5 ms for a sampling step.  Calling this from
        inside a timed loop makes the instrument cost 450x the thing measured --
        a 300-step benchmark checkpointing 10 times charges itself 52.7 ms/step
        of pure measurement, which is how a 3.5 ms/step sampler came to be
        reported at 59.9 and "slower than gapsplit" on Recon2.

        Worse, the cost is NOT symmetric across arms: it scales with record size,
        so an arm carrying a big FVA seed is charged ~50x what an arm holding only
        bound vectors is.  Exclude checkpoint time from any reported ms/step, or
        pass `exact=False` for the value from the last rebuild.
        """
        if exact:
            self._rebuild()
        return 1.0 - float(np.mean(self.mg / self.rng))


class Interpolate:
    """Coverage sampling with zero solves, on a convex polytope.

    `pool0` must be FEASIBLE vectors (FVA solution vectors are ideal and free --
    they cost nothing beyond the FVA you already ran, and on Recon2 1000 of them
    score 0.6144 where 1000 QP solves score 0.3828).  Bound vectors do not belong
    here; they go to `Coverage` as anchors.

    Reaction selection is either round-robin or PROPORTIONAL to a score mixing
    each reaction's free gap gain with its `spread` (how many coordinates its FVA
    path direction moves).  Proportional matters: argmax selection fixates -- an
    early run picked only reactions 0 and 98 out of 7324.
    """

    def __init__(self, P, idx, pool0, spread=None, k=4, poolmax=4000,
                 wspread=0.0, seed=0, policy="cyclic",
                 endpoint="uniform", collapse=False, oracle=None, k_probe=32):
        # `oracle` is the ENTIRE non-convex extension: a callable v -> bool that
        # certifies whatever convexity does not give for free.  None (the default)
        # is the convex case and costs nothing -- see `step`.  For loopless
        # models pass a `bosampler.loopless.LooplessOracle`; `pool0` must then be
        # VERIFIED loopless, since a loopy pool vector poisons every blend drawn
        # from it (`loopless.filter_loopless`).
        #
        # PLAUSIBILITY CAPABILITY (optional, problem-agnostic).  If the oracle
        # additionally exposes
        #
        #     plausibility(ia, ib, pool) -> floats in [0, 1]
        #
        # (vectorised: `ia`, `ib` are equal-length integer arrays of pool ROW
        # indices, one candidate pair per position, `pool` is this sampler's
        # Pool), the value is the oracle's PRIOR that the blend of rows ia[t]
        # and ib[t] would pass the full check.  `step` then (a) forms up to
        # `k_probe` candidate pairs instead of k, (b) ranks candidates by
        # gain * plausibility instead of gain alone, and (c) still submits at
        # most k candidates per step to the full oracle -- the same certificate
        # budget as without the capability.  The capability is a RANKING signal
        # only and never certifies anything: every accepted sample still passes
        # the full oracle.  When the method is absent, `step` is identical to
        # the plain path: same rng draws, same candidate set, same ranking,
        # same checks -- `k_probe` is then unused.
        self.oracle = oracle
        self.k_probe = int(k_probe)
        self.n_rejected = 0
        self.P = P
        self.idx = np.asarray(idx)
        if len(pool0) == 0:
            raise ValueError("pool0 must contain at least one feasible vector")
        self.pool = Pool(pool0, maxlen=poolmax)
        self.k = k
        self.poolmax = poolmax
        self.wspread = wspread
        # "cyclic"   -- gapsplit's own round-robin.  Reaction j gets its turn once
        #               every n steps, so the WORST reaction cannot improve until
        #               then: measured flat at max_gap/range = 1.0 for all 300
        #               steps on iJR904 (666 reactions) and Recon2 (7324).
        # "worstgap" -- sample proportional to each reaction's free gap gain, so
        #               effort goes where the gaps are.  Proportional and NOT
        #               argmax: argmax fixates, and once picked only reactions 0
        #               and 98 out of 7324.  Costs O(n log m) per step from the
        #               vectorised locate we already pay for scoring, ~1-2 ms at
        #               Recon2 against a 69-118 ms step.
        if policy not in ("cyclic", "worstgap"):
            raise ValueError("policy must be 'cyclic' or 'worstgap'")
        self.policy = policy
        if endpoint not in ("uniform", "spread"):
            raise ValueError("endpoint must be 'uniform' or 'spread'")
        self.endpoint = endpoint
        self.rng = np.random.default_rng(seed)
        self.cov = Coverage(self.idx, P.lb[self.idx], P.ub[self.idx])
        for v in self.pool.view():
            self.cov.raw.append(v[self.idx])
        self.cov._rebuild()
        if spread is None:
            self.sp = np.ones(len(self.idx))
        else:
            sp = np.asarray(spread, float)
            self.sp = sp / max(sp.max(), 1e-12)
        # halved whenever a reaction fails to bracket, so effort drains away
        # from reactions the pool cannot reach instead of retrying them forever
        self.penalty = np.ones(len(self.idx))
        # ENZYME SUBSETS: target one representative per fully-coupled set.  The
        # COVERAGE METRIC is deliberately left over all of `idx`, so the numbers
        # stay comparable -- only the targeting collapses.
        if collapse is False:
            self.tgt = np.arange(len(self.idx))
            self.groups = len(self.idx)
        else:
            lab = (np.asarray(collapse) if collapse is not True
                   else enzyme_subsets(P, self.idx))
            if len(lab) != len(self.idx):
                raise ValueError("collapse labels must be one per active reaction")
            _, first = np.unique(lab, return_index=True)
            self.tgt = np.sort(first)
            self.groups = len(self.tgt)
        self.n_interp = 0
        self.n_fail = 0

    def _pick(self, cur):
        T = self.tgt
        if self.policy == "cyclic" and self.wspread == 0.0:
            return int(T[cur % len(T)])
        gg = self.cov.gap_gains()[T]
        gg = gg / max(gg.max(), 1e-30)
        if self.policy == "worstgap" and self.wspread == 0.0:
            score = gg * self.penalty[T]
        else:
            score = (gg ** (1.0 - self.wspread)) * (self.sp[T] ** self.wspread) \
                * self.penalty[T]
        tot = score.sum()
        if tot <= 0 or not np.isfinite(tot):
            return int(T[cur % len(T)])
        return int(T[self.rng.choice(len(T), p=score / tot)])

    def step(self):
        a = self._pick(self.n_interp + self.n_fail)
        j = int(self.idx[a])
        tau = self.cov.target_of(a)
        col = self.pool.col(j)                     # O(n) view, never a copy
        below = np.flatnonzero(col < tau - 1e-12)
        above = np.flatnonzero(col > tau + 1e-12)
        if below.size == 0 or above.size == 0:
            self.n_fail += 1
            self.penalty[a] *= 0.5
            return None
        # With a plausibility capability (see __init__), probe a wider candidate
        # set -- widening costs only gain scoring, and the capability's ranking
        # decides which few candidates are worth the oracle's budget.  Without
        # it, kk == self.k and the draws below are bit-identical to the plain
        # path.
        plaus = getattr(self.oracle, "plausibility", None)
        kk = self.k if plaus is None else max(self.k, self.k_probe)
        ia = self.rng.choice(below, size=min(kk, below.size))
        ib = self.rng.choice(above, size=min(kk, above.size))
        if self.endpoint == "spread" and len(ia) > 1 and len(ib) > 1:
            # ENDPOINT BIAS.  v = lam A + (1-lam) B has v_i strictly between A_i
            # and B_i for EVERY i, so the coverage this step buys outside the
            # target reaction is bounded by how much A and B differ elsewhere.
            # A pair that agrees everywhere except coordinate j yields a blend
            # nearly identical to both endpoints: it fills the gap in j and
            # contributes nothing else.  Since 72% of gapsplit's coverage gain is
            # INCIDENTAL, that is most of the value being thrown away by pairing
            # candidates positionally at random.
            #
            # Rank the K x K pairs by distance weighted by each reaction's
            # CURRENT gap -- distance in a reaction already well covered is worth
            # nothing.  Costs K^2 * n_active flops (~0.5M at K=8, n=7324).
            A = self.pool.buf[ia][:, self.idx]
            B = self.pool.buf[ib][:, self.idx]
            wgt = self.cov.mg / self.cov.rng
            D = np.abs(A[:, None, :] - B[None, :, :]) @ (wgt / self.cov.rng)
            order = np.dstack(np.unravel_index(
                np.argsort(D, axis=None)[::-1][:kk], D.shape))[0]
            ia, ib = ia[order[:, 0]], ib[order[:, 1]]
        cands = []
        pairs = []                     # pool row indices behind each candidate
        for t in range(min(len(ia), len(ib))):
            A, B = self.pool[ia[t]], self.pool[ib[t]]
            den = A[j] - B[j]
            if abs(den) < 1e-12:
                continue
            lam = (tau - B[j]) / den
            if not (0.0 <= lam <= 1.0):
                continue
            v = lam * A + (1.0 - lam) * B
            cands.append((self.cov.gain_of(v), v))
            pairs.append((int(ia[t]), int(ib[t])))
        if not cands:
            self.n_fail += 1
            self.penalty[a] *= 0.5
            return None

        # THE ONLY CONVEX/NON-CONVEX DIFFERENCE IS `self.oracle`.
        #
        # Convex (oracle is None): the first candidate in descending gain order IS
        # argmax gain, so this is exactly the old `best/bestg` loop -- Python's
        # sort is stable, so even the tie-break (first maximum encountered) is
        # preserved.  Zero solves, and the sort is O(k log k) at k <= 8.
        #
        # Non-convex: a blend can land on a sign pattern neither endpoint had, so
        # the best-gain candidate may be infeasible and we walk down the ranking.
        # Ranking BEFORE consulting the oracle is what makes each oracle call buy
        # the most coverage available; checking in arbitrary order spends the same
        # budget on worse samples.
        #
        # With a plausibility capability the ranking key becomes
        # gain * plausibility(pair): the oracle's own prior reorders the queue so
        # the (still at most k) full checks land on candidates likely to survive.
        # Plausibility never decides membership -- acceptance below is the full
        # oracle's verdict on every sample, capability or not.
        if plaus is None:
            order = sorted(range(len(cands)), key=lambda t: -cands[t][0])
        else:
            w = np.asarray(plaus(
                np.fromiter((p[0] for p in pairs), np.int64, len(pairs)),
                np.fromiter((p[1] for p in pairs), np.int64, len(pairs)),
                self.pool), float)
            order = sorted(range(len(cands)),
                           key=lambda t: -(cands[t][0] * float(w[t])))
        for t in order[:self.k]:
            v = cands[t][1]
            if self.oracle is None or self.oracle(v):
                self.cov.add(v)
                self.pool.append(v)                # Pool enforces maxlen itself
                self.n_interp += 1
                return v
            self.n_rejected += 1
        self.n_fail += 1
        self.penalty[a] *= 0.5
        return None

    def run(self, budget=None, n=None):
        """Run for `budget` seconds or until `n` samples. Returns a dict."""
        if (budget is None) == (n is None):
            raise ValueError("give exactly one of budget= or n=")
        t0 = time.time()
        while (time.time() - t0 < budget) if budget else (self.n_interp < n):
            self.step()
        out = {"coverage": self.cov.coverage(), "interpolated": self.n_interp,
               "unbracketed": self.n_fail, "rejected": self.n_rejected,
               "solves": 0, "seconds": time.time() - t0, "pool": len(self.pool)}
        if self.oracle is not None:
            out["solves"] = getattr(self.oracle, "solves", 0)
            out["oracle"] = dict(getattr(self.oracle, "stats", {}))
        return out


def enzyme_subsets(P, idx, tol=1e-8):
    """Group reactions whose fluxes are in fixed ratio in EVERY steady state.

    Every feasible point is v = v0 + N z, so v_i = v0_i + N_i . z.  If rows i and
    k of N are parallel, N_i = alpha N_k, then v_i = alpha v_k + c EXACTLY, for
    all z.  Such a group is an "enzyme subset" (Pfeiffer et al. 1999) -- an
    unbranched pathway segment with no alternative route in or out, so the
    reactions operate as one unit and knocking out any member blocks the rest.
    It is the solver-free core of flux coupling analysis, which needs an LP per
    pair to find the partial couplings this misses.

    Why targeting cares: coverage is max_gap_j / range_j per reaction, and an
    exact affine map carries gaps to gaps.  Reaction i's largest gap IS reaction
    k's largest gap, so landing on one lands on the other, and giving both a turn
    in the round-robin spends the second turn on nothing.  Measured redundancy is
    36-40% of active reactions on e_coli_core, iJR904 and iND750, verified against
    uniform CDHR samples at |corr| = 1.0000 inside every group.

    Rows with no null-space component are reactions stoichiometry pins to a single
    value; their gap is already zero.  They are put in one group together rather
    than each taking a turn.

    Returns an integer label per entry of `idx`.
    """
    N = getattr(P, "N", None)
    if N is None:
        raise ValueError("enzyme_subsets needs P.N (a null-space basis); the "
                         "Lite duck-type does not -- pass labels explicitly")
    idx = np.asarray(idx)
    M = np.asarray(N)[idx]
    nrm = np.linalg.norm(M, axis=1)
    free = nrm > tol * max(nrm.max(), 1.0)
    lab = np.full(len(idx), -1, dtype=np.int64)      # -1 = the pinned group
    if free.any():
        U = M[free] / nrm[free][:, None]
        par = sparse.csr_matrix(np.abs(U @ U.T) > 1.0 - tol)
        lab[free] = csgraph.connected_components(par, directed=False)[1]
    return lab


def fva_seed(P, active, lp_env=None, check=True):
    """FVA over every active reaction, keeping the whole solution VECTOR.

    Ordinary FVA reports min/max per reaction and throws the solution away --
    two numbers out of a d-dimensional point.  Kept instead, the 2n vectors are
    the interpolation pool, and they are free: you ran the FVA for the bounds.
    Measured on Recon2: 1000 of these score coverage 0.6144 against 0.3828 for
    1000 actual QP solves.

    THERE IS NO CAP, deliberately.  An earlier version stopped after 1200
    vectors, which is 600 of iJR904's 666 reactions, and the truncation is by
    ITERATION ORDER -- so it silently dropped the last 66 reactions' extremes.
    Every vector here is load-bearing: the pool must contain argmin and argmax
    of each reaction, because that is exactly the condition for

        lam = (tau - B_j) / (A_j - B_j)   to have a solution in (0, 1)

    for any target tau in that reaction's range.  Without both, the reaction's
    outermost bins have no bracketing pair, are unreachable forever, and a
    bottleneck stopping rule will spin on them until it hits its sample cap.
    There is nothing to economise on -- capping does not trade accuracy for
    speed, it removes reachability from a subset of coordinates.

    `check=True` verifies coverage and raises rather than letting it pass
    silently, which is how the original bug survived four benchmarks.
    """
    import gurobipy as gp
    from scipy import sparse

    env = lp_env or gp.Env(params={"OutputFlag": 0, "FeasibilityTol": 1e-9})
    lp = gp.Model(env=env)
    x = lp.addMVar(len(P.lb), lb=P.lb, ub=P.ub)
    lp.addConstr(sparse.csr_matrix(P.S) @ x == P.b)
    out, covered = [], []
    for j in active:
        got = 0
        for sense in (gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE):
            lp.setObjective(x[int(j)], sense)
            lp.optimize()
            if lp.Status == gp.GRB.OPTIMAL:
                out.append(np.array(x.X))
                got += 1
        covered.append(got == 2)
    if check and not all(covered):
        miss = int(len(covered) - sum(covered))
        raise RuntimeError(
            f"FVA seed covers only {sum(covered)}/{len(active)} reactions; "
            f"{miss} lack a bracketing pair and their extremes are unreachable")
    return out


def validity(P, V, tol=1e-6):
    """Fraction of V satisfying Sv = b and lb <= v <= ub. Report this always."""
    A = np.asarray(V)
    r = np.abs(A @ P.S.T - P.b).max(axis=1)
    box = (A >= P.lb - tol).all(axis=1) & (A <= P.ub + tol).all(axis=1)
    return float(np.mean((r < tol) & box))

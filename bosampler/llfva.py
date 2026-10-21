"""Loopless FVA driver: the package's own capped/uncapped ll-FVA.

The division of labor in BOSampler's pipeline:

  vanilla FVA (+ solutions)   cobrapy PR #1482 via `bosampler.fva`
  loopless FVA (+ solutions)  THIS MODULE -- a native-Gurobi driver, because
                              the semantics it needs (per-solve TimeLimit,
                              keeping timeout incumbents, MIP-start chains)
                              are not expressible through stock cobra/optlang

Design, as measured on Recon2:
  - all 2n directions solved as MILPs over add_loopless(method="potentials"),
    so every harvested vector is a certified loopless flux (the seed)
  - warm-start chain: one objective-0 SolutionLimit=1 solve, then each
    solve passes the previous solution as a MIP start -- no solve ever lacks
    an incumbent (0 empty in 14,648 at TL=4)
  - tl > 0: cap each solve (58% prove at 4 s; 68 min on 8x2 cores)
    tl = 0: uncapped -- the tail-measurement mode; may run for days
  - stride assignment spreads the alphabetical hard clusters evenly --
    but lanes cannot steal work, so a lane that draws the monsters can
    strand its tail: resume such a campaign with pending="scan", which
    re-strides the unfinished lane tails over fresh workers (measured on
    recon2: 14/16 lanes finished while 2 held 616 reactions)
  - per-worker checkpoint + resume (preemption loses <= ckpt reactions)
  - per-solve timing CSV (reaction, sense, seconds, status, objective)
  - bounds in the merged product are WITNESS bounds -- lo = V.min(0),
    hi = V.max(0) -- sound inner bounds, exact for proven directions; the
    dual-bound alternative was measured unsound under big-M numerics
"""
import os
import time

import numpy as np

_LOADERS = ("recon2",)


def _load(loader, growth_fraction):
    if callable(loader):               # any picklable zero-arg model factory
        m = loader()
    elif loader == "recon2":
        from bosampler.models import load_recon2
        m = load_recon2(max_bound=1000.0)
    else:
        raise ValueError(f"unknown loader {loader!r}; pass a callable or add "
                         "it to bosampler.llfva")
    if growth_fraction:
        opt = m.slim_optimize()
        bio = str(m.objective.expression).split("*")[1].split()[0]
        m.reactions.get_by_id(bio).lower_bound = growth_fraction * opt
    return m


def _worker(args):
    (w, nproc, loader, growth, tl, threads, ckpt, out_dir, tag,
     rxns, sub) = args
    import gurobipy as gp
    from cobra.flux_analysis.loopless import add_loopless

    stem = f"llfva_{tag}_{sub}" if sub else f"llfva_{tag}"
    part = os.path.join(out_dir, f"{stem}_part{w}.npz")
    tcsv = os.path.join(out_dir, f"{stem}_times_{w}.csv")
    done, vecs, opts = 0, [], 0
    if os.path.exists(part):
        z = np.load(part)
        done, opts = int(z["done"]), int(z["opts"])
        vecs = list(z["V"])
        print(f"  w{w}: RESUMING at reaction {done} "
              f"({len(vecs)} vectors banked)", flush=True)

    m = _load(loader, growth)
    with m as mm:
        add_loopless(mm, method="potentials")
        mm.objective = mm.problem.Objective(0, direction="min")
        g = mm.solver.problem
        g.Params.OutputFlag = 0
        g.Params.Threads = threads
        mm.solver.update()
        vmap = {v.VarName: v for v in g.getVars()}
        fw = [vmap[r.forward_variable.name] for r in mm.reactions]
        rv = [vmap[r.reverse_variable.name] for r in mm.reactions]
        allv = g.getVars()

        g.Params.SolutionLimit = 1
        g.optimize()
        g.Params.SolutionLimit = 2000000000
        last = g.getAttr("X", allv) if g.SolCount else None
        if tl > 0:
            g.Params.TimeLimit = tl

        rids = [r.id for r in mm.reactions]
        js = list(rxns) if rxns is not None else list(range(w, len(rids),
                                                           nproc))
        t0 = time.time()
        with open(tcsv, "a") as th:
            for k in range(done, len(js)):
                j = js[k]
                for sense in (gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE):
                    g.setObjective(fw[j] - rv[j], sense)
                    if last is not None:
                        g.setAttr("Start", allv, last)
                    t1 = time.time()
                    g.optimize()
                    dt = time.time() - t1
                    st = {2: "OPT", 9: "TL"}.get(g.Status, str(g.Status))
                    ob = g.ObjVal if g.SolCount else float("nan")
                    th.write(f"{rids[j]},{'max' if sense == -1 else 'min'},"
                             f"{dt:.2f},{st},{ob:.6g}\n")
                    th.flush()
                    if g.SolCount:
                        vecs.append(np.array(g.getAttr("X", fw))
                                    - np.array(g.getAttr("X", rv)))
                        last = g.getAttr("X", allv)
                        opts += int(g.Status == gp.GRB.OPTIMAL)
                if (k + 1) % ckpt == 0 or k == len(js) - 1:
                    np.savez_compressed(part, V=np.array(vecs), done=k + 1,
                                        opts=opts)
                if (k + 1) % 100 == 0:
                    el = time.time() - t0
                    print(f"  w{w}: {k+1}/{len(js)} rxns  {len(vecs)} vecs  "
                          f"{opts} optimal  {el/(k+1-done):.2f}s/rxn",
                          flush=True)
    return w


def loopless_fva(loader="recon2", growth_fraction=0.3, tl=4.0, nproc=8,
                 threads=2, ckpt=20, out_dir="data", tag=None,
                 keep_parts=False, pending=None, src_nproc=None,
                 run_label="relane"):
    """Run the driver; return the merged npz path.

    loader is "recon2" or any picklable zero-arg callable returning a
    prepared cobra Model (workers are spawned, so a module-level function or
    a functools.partial of one); pass tag= for callables.

    pending selects the work assignment:
      None          the full model, static lanes (reaction index == w mod
                    nproc), each lane resuming from its own checkpoint --
                    the standard fresh-campaign mode
      "scan"        resume a lane campaign that ran out of walltime: read
                    the base lane checkpoints (written with src_nproc
                    workers; defaults to nproc) and RE-STRIDE the union of
                    unfinished lane tails across nproc fresh workers.
                    Exists because static lanes cannot steal work: on
                    recon2, alphabetical enzyme families clustered the hard
                    MILPs into two lanes that held 616 reactions while 14
                    lanes sat finished.  Scan the BASE lanes only; re-scans
                    after a partial relane must reuse the SAME nproc (the
                    relane checkpoints resume positionally) or a fresh
                    run_label.
      [i, j, ...]   an explicit reaction-index list -- the surgical mode
                    (stragglers, monster re-solves, smoke tests)
    Relane/pending part files are namespaced llfva_{tag}_{run_label}_part*.

    Product: llfva_{tag}.npz -- seed (deduplicated certified loopless
    vectors), witness lo/hi, n_optimal, n_solves, seconds -- merged across
    ALL part families of the tag.
    """
    import glob
    name = loader if isinstance(loader, str) else getattr(loader, "__name__",
                                                            "model")
    tag = tag or (f"{name}_tl{tl:g}" if tl > 0 else f"{name}_uncapped")
    os.makedirs(out_dir, exist_ok=True)
    from multiprocessing import get_context
    t0 = time.time()

    if pending == "scan":
        m = _load(loader, 0)  # lane math only needs the reaction count
        n = len(m.reactions)
        del m
        src = src_nproc or nproc
        pending = []
        for w in range(src):
            lane = list(range(w, n, src))
            p = os.path.join(out_dir, f"llfva_{tag}_part{w}.npz")
            done = int(np.load(p)["done"]) if os.path.exists(p) else 0
            pending.extend(lane[done:])
            if done < len(lane):
                print(f"  lane {w}: {done}/{len(lane)} done, "
                      f"{len(lane) - done} pending", flush=True)
        pending.sort()

    if pending is not None:
        jobs = [(w, nproc, loader, growth_fraction, tl, threads, ckpt,
                 out_dir, tag, pending[w::nproc], run_label)
                for w in range(nproc)]
        print(f"llfva driver [{run_label}]: {len(pending)} pending reactions "
              f"over {nproc} workers x {threads} threads, "
              f"TL={'none' if tl <= 0 else tl}, tag={tag}", flush=True)
    else:
        jobs = [(w, nproc, loader, growth_fraction, tl, threads, ckpt,
                 out_dir, tag, None, "") for w in range(nproc)]
        print(f"llfva driver: loader={loader}, {nproc} workers x {threads} "
              f"threads, TL={'none' if tl <= 0 else tl}, ckpt={ckpt}, "
              f"tag={tag}", flush=True)

    with get_context("spawn").Pool(nproc) as pool:
        pool.map(_worker, jobs)

    V, n_opt = [], 0
    parts = sorted(glob.glob(os.path.join(out_dir, f"llfva_{tag}*part*.npz")))
    for p in parts:
        z = np.load(p)
        if len(z["V"]):
            V.append(z["V"])
        n_opt += int(z["opts"])
    V = np.vstack(V)
    lo, hi = V.min(axis=0), V.max(axis=0)
    Vd = np.unique(np.round(V, 9), axis=0)
    el = time.time() - t0
    outp = os.path.join(out_dir, f"llfva_{tag}.npz")
    np.savez_compressed(outp, seed=Vd, lo=lo, hi=hi, n_optimal=n_opt,
                        n_solves=2 * len(lo), seconds=el)
    if not keep_parts:
        for p in parts:
            os.remove(p)
    print(f"LLFVA DONE in {el/60:.0f}m: {2*len(lo)} solves, {n_opt} proven "
          f"({100*n_opt/(2*len(lo)):.0f}%), {len(V)} vectors "
          f"({len(Vd)} unique) -> {outp}", flush=True)
    return outp

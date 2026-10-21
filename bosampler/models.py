"""Model loading, for the one model cobra cannot read.

e_coli_core, iJR904 and iND750 all come from `cobra.io.load_model`. Recon 2.05
does not: `cobra.io.load_matlab_model` raises

    TypeError: only 0-dimensional arrays can be converted to Python scalars

in `mat_parse_notes`, after failing to parse a malformed gene-reaction rule. That
has been true across every session in this project, which is why Recon2 was
previously handled as a raw polytope (S, lb, ub, b) rather than a cobra Model --
and why it did not fit the shared FVA path that every other model uses.

This reads the arrays with scipy and assembles the Model itself, skipping notes
and GPRs entirely. Those fields are irrelevant to sampling: the polytope is
determined by S, lb, ub and the objective.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECON2 = os.path.join(HERE, "..", "models", "Recon205_20150515Consistent.mat")


def load_recon2(path=RECON2, max_bound=None):
    """Recon 2.05 consistent as a cobra Model, bypassing cobra's .mat reader.

    `max_bound` clamps the box HERE, on the raw arrays, before any Reaction
    exists. This is TIDINESS, not necessity: a per-reaction clamp afterwards
    measures 0.01 ms per assignment -- 0.2 s for all 14648 on this model, linear,
    with or without a live solver. An earlier note in this project claimed that
    loop cost over 15 minutes and was quadratic; it does not reproduce and the
    figure was wrong by ~3000x. the shared load_model prep in benchmarks/ would be fine here.

    The reason this loader exists is unrelated and unchanged: cobra cannot read
    the .mat file at all.
    """
    import cobra
    from scipy.io import loadmat

    raw = loadmat(path, struct_as_record=False, squeeze_me=True)
    key = [k for k in raw if not k.startswith("__")][0]
    m = raw[key]

    S = np.asarray(m.S.todense() if hasattr(m.S, "todense") else m.S, float)
    lb = np.asarray(m.lb, float).ravel()
    ub = np.asarray(m.ub, float).ravel()
    c = np.asarray(m.c, float).ravel()
    if max_bound is not None:
        lb = np.maximum(lb, -max_bound)
        ub = np.minimum(ub, max_bound)
    rxns = [str(x) for x in np.asarray(m.rxns).ravel()]
    mets = [str(x) for x in np.asarray(m.mets).ravel()]

    model = cobra.Model(str(getattr(m, "description", "Recon2")))
    model.add_metabolites([cobra.Metabolite(i) for i in mets])
    reactions = []
    for j, rid in enumerate(rxns):
        r = cobra.Reaction(rid)
        r.lower_bound, r.upper_bound = float(lb[j]), float(ub[j])
        reactions.append(r)
    model.add_reactions(reactions)
    for j, r in enumerate(reactions):
        nz = np.flatnonzero(S[:, j])
        if len(nz):
            r.add_metabolites({model.metabolites[int(i)]: float(S[i, j])
                               for i in nz})
    obj = np.flatnonzero(c)
    if len(obj):
        model.objective = {reactions[int(j)]: float(c[j]) for j in obj}
    return model

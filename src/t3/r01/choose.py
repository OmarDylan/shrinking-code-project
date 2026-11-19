import sys, json, csv, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BUDGET, K = 10, 3

def read(path):
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [[c.strip() for c in row] for row in r[1:] if row]

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def parse_goals(resp, gnames):
    skip = {"index", "row", "id", "inspect"}
    g = None
    if isinstance(resp, dict):
        if all(n in resp for n in gnames):
            g = [resp[n] for n in gnames]
        else:
            for k in ("goals", "values", "observed", "y"):
                if k in resp:
                    g = resp[k]
                    break
            if g is None:
                g = [v for k, v in resp.items() if k not in skip]
            if isinstance(g, dict):
                g = [g[n] for n in gnames] if all(n in g for n in gnames) else list(g.values())
    elif isinstance(resp, list):
        g = resp[1:] if len(resp) == len(gnames) + 1 else resp
    return [float(x) for x in g][:len(gnames)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    h, tr = read(a.train)
    th, te = read(a.test)
    gcols = [i for i, n in enumerate(h) if n[-1] in "+-"]
    gnames = [h[i] for i in gcols]
    fcols = [i for i, n in enumerate(h) if n[-1] not in "+-X"]
    tidx = {n: i for i, n in enumerate(th)}
    fnames = [h[i] for i in fcols]

    # goals normalization from train
    tr = [r for r in tr if all(num(r[i]) is not None for i in gcols)]
    G = np.array([[float(r[i]) for i in gcols] for r in tr])
    gmin, gmax = G.min(0), G.max(0)
    gr = np.where(gmax > gmin, gmax - gmin, 1.0)
    maxi = np.array([h[i][-1] == "+" for i in gcols])

    def d2h(g):
        n = np.clip((np.asarray(g, float) - gmin) / gr, 0, 1)
        n = np.where(maxi, 1 - n, n)
        return float(np.sqrt((n ** 2).mean()))

    def col(rows, hdr_idx, name):
        j = hdr_idx.get(name)
        return [r[j] if j is not None and j < len(r) else "?" for r in rows]

    hidx = {n: i for i, n in enumerate(h)}
    blocks_tr, blocks_te = [], []
    for n in fnames:
        a_, b_ = col(tr, hidx, n), col(te, tidx, n)
        if n[0].isupper():
            tv = [num(v) for v in a_]
            tv = np.array([v for v in tv if v is not None])
            lo, hi = tv.min(), tv.max()
            sp = hi - lo if hi > lo else 1.0
            med = (np.median(tv) - lo) / sp
            f = lambda L: [(num(v) - lo) / sp if num(v) is not None else med for v in L]
            blocks_tr.append(np.array(f(a_))[:, None])
            blocks_te.append(np.array(f(b_))[:, None])
        else:
            cats = sorted(set(a_) | set(b_))
            f = lambda L: np.array([[1.0 if v == c else 0.0 for c in cats] for v in L])
            blocks_tr.append(f(a_))
            blocks_te.append(f(b_))
    Xtr = np.hstack(blocks_tr)
    Xte = np.hstack(blocks_te)
    ytr = np.array([d2h(g) for g in G])

    n = len(te)
    budget = min(BUDGET, n)
    obs = {}

    def inspect(i):
        print(json.dumps({"inspect": int(i)}), flush=True)
        line = sys.stdin.readline()
        obs[i] = d2h(parse_goals(json.loads(line), gnames))

    def fit_pick(cnt, kappa):
        X, y = Xtr, ytr
        if obs:
            ids = list(obs)
            X = np.vstack([Xtr] + [Xte[ids]] * 3)
            y = np.concatenate([ytr] + [np.array([obs[i] for i in ids])] * 3)
        m = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=1).fit(X, y)
        P = np.array([t.predict(Xte) for t in m.estimators_])
        s = P.mean(0) - kappa * P.std(0)
        order = [i for i in np.argsort(s, kind="stable") if i not in obs]
        return order[:cnt]

    for i in fit_pick(5, 0.0):
        if len(obs) < budget:
            inspect(int(i))
    while len(obs) < budget:
        inspect(int(fit_pick(1, 0.5)[0]))

    sel = sorted(obs, key=lambda i: (obs[i], i))[:K]
    print(json.dumps({"selected": [int(i) for i in sel]}), flush=True)

main()

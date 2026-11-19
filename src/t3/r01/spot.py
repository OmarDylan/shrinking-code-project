import sys, json, csv, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return [h.strip() for h in rows[0]], [r for r in rows[1:] if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    th, tr = load(a.train)
    eh, te = load(a.test)
    cols = []
    for h in th:
        if h[-1] in "+-X" or h not in eh:
            continue
        cols.append((h, h[0].isupper()))
    ti = {h: i for i, h in enumerate(th)}
    ei = {h: i for i, h in enumerate(eh)}

    def get(rows, idx, h):
        out = []
        for r in rows:
            v = r[idx[h]].strip() if idx[h] < len(r) else "?"
            out.append(v)
        return out

    def tofloat(v):
        try:
            return float(v)
        except ValueError:
            return np.nan

    num_tr, num_te, cat_tr, cat_te = [], [], [], []
    names_n, names_c = [], []
    for h, isnum in cols:
        a_, b_ = get(tr, ti, h), get(te, ei, h)
        if isnum:
            names_n.append(h)
            num_tr.append([tofloat(v) for v in a_])
            num_te.append([tofloat(v) for v in b_])
        else:
            names_c.append(h)
            cat_tr.append(a_)
            cat_te.append(b_)
    n_tr, n_te = len(tr), len(te)
    Xn = np.array(num_tr, float).T.reshape(n_tr, -1)
    Tn = np.array(num_te, float).T.reshape(n_te, -1)

    med = np.nanmedian(Xn, axis=0) if Xn.shape[1] else np.array([])
    sc = []
    for j in range(Xn.shape[1]):
        c = Xn[:, j][~np.isnan(Xn[:, j])]
        q1, q3 = np.percentile(c, [25, 75])
        s = (q3 - q1) / 1.349
        if s < 1e-9:
            s = c.std()
        if s < 1e-9:
            s = 1.0
        sc.append(s)
    sc = np.array(sc)

    def numz(M):
        M = M.copy()
        for j in range(M.shape[1]):
            M[np.isnan(M[:, j]), j] = med[j]
        return (M - med) / sc

    Zn, Zt = numz(Xn), numz(Tn)

    # categorical
    cat_feats_tr, cat_feats_te = [], []
    rare_tr = np.zeros(n_tr)
    rare_te = np.zeros(n_te)
    for c in range(len(names_c)):
        vals = [v for v in cat_tr[c] if v != "?"]
        cats = sorted(set(vals))
        freq = {k: vals.count(k) / max(1, len(vals)) for k in cats}
        mode = max(freq, key=freq.get) if freq else ""

        def enc(lst):
            lst = [mode if v == "?" else v for v in lst]
            oh = np.array([[1.0 if v == k else 0.0 for k in cats] for v in lst]).reshape(len(lst), len(cats))
            rare = np.array([4.0 if freq.get(v, 0) < 0.02 else 0.0 for v in lst])
            return oh, rare

        o1, r1 = enc(cat_tr[c])
        o2, r2 = enc(cat_te[c])
        cat_feats_tr.append(o1)
        cat_feats_te.append(o2)
        rare_tr = np.maximum(rare_tr, r1)
        rare_te = np.maximum(rare_te, r2)

    Ftr = np.hstack([Zn] + cat_feats_tr) if (Zn.shape[1] or cat_feats_tr) else np.zeros((n_tr, 1))
    Fte = np.hstack([Zt] + cat_feats_te) if (Zt.shape[1] or cat_feats_te) else np.zeros((n_te, 1))

    k = max(1, min(5, n_tr - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Ftr)
    d_tr = nn.kneighbors(Ftr)[0][:, 1:].mean(axis=1)
    d_te = nn.kneighbors(Fte, n_neighbors=k)[0].mean(axis=1)

    def maxz(Z):
        return np.abs(Z).max(axis=1) if Z.shape[1] else np.zeros(len(Z))

    raw_tr = d_tr + 0.5 * np.maximum(maxz(Zn), rare_tr)
    raw_te = d_te + 0.5 * np.maximum(maxz(Zt), rare_te)

    m = float(np.median(raw_tr)) or 1e-6

    def squash(r):
        return r / (r + m)

    thr = float(squash(np.quantile(raw_tr, 0.93)))
    scores = squash(raw_te)
    out = {
        "scores": [float(x) for x in scores],
        "alerts": [bool(x > thr) for x in scores],
        "threshold": thr,
    }
    print(json.dumps(out))


main()

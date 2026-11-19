import sys, csv, json
import numpy as np
from sklearn.ensemble import IsolationForest

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    X = np.array([[float(row[c]) if row[c].strip() not in ("?", "") else np.nan
                   for c in NUM] for row in rows]).reshape(len(rows), len(NUM))
    o = [(row[CAT].strip() if row[CAT].strip() not in ("?", "") else None) for row in rows]
    return X, o


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    Xt, ot = load(tr)
    Xe, oe = load(te)
    med = np.nanmedian(Xt, axis=0)
    med = np.where(np.isnan(med), 0, med)
    Xt = np.where(np.isnan(Xt), med, Xt)
    mad = 1.4826 * np.median(np.abs(Xt - med), axis=0)
    sd = Xt.std(axis=0)
    sc = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
    cats = sorted({c for c in ot if c is not None})

    def feats(X, o):
        X = np.where(np.isnan(X), med, X)
        Z = (X - med) / sc
        oh = np.array([[1.0 if c == k else 0.0 for k in cats] for c in o]).reshape(len(o), len(cats))
        return Z, np.hstack([Z, oh])

    Zt, Ft = feats(Xt, ot)
    Ze, Fe = feats(Xe, oe)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Ft)
    it = -iso.score_samples(Ft)
    ie = -iso.score_samples(Fe)
    zt = np.abs(Zt).max(axis=1)
    ze = np.abs(Ze).max(axis=1)

    n = len(Xt)

    def ecdf(ref):
        s = np.sort(ref)
        return lambda v: np.searchsorted(s, v, side="right") / (n + 1.0)

    f1, f2 = ecdf(zt), ecdf(it)
    rt = np.maximum(f1(zt), f2(it))
    re = np.maximum(f1(ze), f2(ie))
    f3 = ecdf(rt)
    st = f3(rt)
    se = f3(re)
    unseen = np.array([c is not None and c not in cats for c in oe], dtype=bool)
    se = np.where(unseen, 1.0, se)
    se = np.clip(se, 0.0, 1.0)
    thr = float(min(0.97, max(0.0, np.quantile(st, 0.97))))
    thr = 0.97
    out = {"scores": [float(x) for x in se],
           "alerts": [bool(x > thr) for x in se],
           "threshold": thr}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()

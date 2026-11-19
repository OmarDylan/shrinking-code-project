import sys, csv, json
import numpy as np


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        rows = [x for x in r if x]
    idx = {n.rstrip("+-X"): i for i, n in enumerate(h)}
    num, org = [], []
    for x in rows:
        v = []
        for c in ("Clndrs", "Volume", "Model"):
            try:
                t = x[idx[c]].strip()
                v.append(np.nan if t in ("?", "") else float(t))
            except Exception:
                v.append(np.nan)
        num.append(v)
        try:
            o = x[idx["origin"]].strip()
        except Exception:
            o = "?"
        org.append(o or "?")
    return np.array(num, float).reshape(-1, 3), org


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    Xtr, otr = load(tr)
    Xte, ote = load(te)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    mad = np.nanmedian(np.abs(Xtr - med), axis=0) * 1.4826
    sd = np.nanstd(Xtr, axis=0)
    rs = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
    sc = np.where(sd > 1e-9, sd, 1.0)
    cats = sorted(set(otr))

    def prep(X, o):
        Xf = np.where(np.isnan(X), med, X)
        Z = (Xf - med) / sc
        oh = np.array([[1.0 if c == v else 0.0 for c in cats] for v in o]).reshape(len(o), len(cats))
        F = np.hstack([Z, oh * 0.7071])
        mz = np.abs((Xf - med) / rs)
        mz = np.where(np.isnan(X), 0.0, mz).max(axis=1) if len(X) else np.zeros(0)
        # rare/unseen category penalty
        freq = {c: otr.count(c) / len(otr) for c in cats}
        rare = np.array([1.0 - min(1.0, freq.get(v, 0.0) * 5) for v in o])
        return F, np.minimum(mz, 20.0), rare

    Ftr, mtr, rtr = prep(Xtr, otr)
    Fte, mte, rte = prep(Xte, ote)
    k = min(5, max(1, len(Ftr) - 1))

    def knn(Fq, loo):
        if len(Fq) == 0:
            return np.zeros(0)
        d = np.sqrt(((Fq[:, None, :] - Ftr[None, :, :]) ** 2).sum(-1))
        d.sort(axis=1)
        if loo:
            d = d[:, 1:k + 1]
        else:
            d = d[:, :k]
        return d.mean(axis=1)

    raw_tr = knn(Ftr, True) + 0.3 * mtr + 1.0 * rtr
    raw_te = knn(Fte, False) + 0.3 * mte + 1.0 * rte
    s = np.sort(raw_tr)

    def ecdf(v):
        return np.searchsorted(s, v, side="right") / len(s)

    sco = np.clip(ecdf(raw_te), 0.0, 1.0)
    thr = 0.95
    out = {
        "scores": [float(x) for x in sco],
        "alerts": [bool(x > thr) for x in sco],
        "threshold": thr,
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()

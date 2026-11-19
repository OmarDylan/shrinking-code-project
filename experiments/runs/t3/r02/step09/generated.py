import sys, csv, json
import numpy as np

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    Xtr = np.array([[num(r.get(c)) for c in NUM] for r in tr])
    Xte = np.array([[num(r.get(c)) for c in NUM] for r in te])
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0, med)
    for X in (Xtr, Xte):
        idx = np.where(np.isnan(X))
        X[idx] = med[idx[1]]
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=1) if len(Xtr) > 1 else np.ones(3)
    sd = np.where(sd < 1e-9, 1.0, sd)
    cats = sorted({r.get(CAT, "?") for r in tr})

    def enc(X, rows):
        Z = (X - mu) / sd
        O = np.array([[1.0 if r.get(CAT, "?") == c else 0.0 for c in cats] for r in rows]).reshape(len(rows), len(cats))
        return Z, O

    Ztr, Otr = enc(Xtr, tr)
    Zte, Ote = enc(Xte, te)
    F = lambda Z, O: np.hstack([Z, O])
    Ftr, Fte = F(Ztr, Otr), F(Zte, Ote)
    n = len(Ftr)
    k = max(1, min(5, n - 1))

    def knn(D):
        D = np.sort(D, axis=1)[:, :k]
        return D.mean(1)

    def dist(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))

    Dtr = dist(Ftr, Ftr)
    np.fill_diagonal(Dtr, np.inf)
    ktr = knn(Dtr)
    kte = knn(dist(Fte, Ftr)) if len(Fte) else np.zeros(0)
    ztr = np.abs(Ztr).max(1)
    zte = np.abs(Zte).max(1) if len(Zte) else np.zeros(0)
    # unseen category penalty in z-stat
    sk, sz = np.sort(ktr), np.sort(ztr)

    def ecdf(s, v):
        return np.searchsorted(s, v, side="right") / len(s)

    str_ = np.maximum(ecdf(sk, ktr), ecdf(sz, ztr))
    ste = np.maximum(ecdf(sk, kte), ecdf(sz, zte))
    thr = float(np.clip(np.quantile(str_, 0.95), 0, 1))
    ste = np.clip(ste, 0, 1)
    print(json.dumps({"scores": [float(x) for x in ste],
                      "alerts": [bool(x > thr) for x in ste],
                      "threshold": thr}))


if __name__ == "__main__":
    main()

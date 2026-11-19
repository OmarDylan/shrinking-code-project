import csv, sys, json, argparse
import numpy as np

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.full((len(rows), len(NUM)), np.nan)
    cat = []
    for i, r in enumerate(rows):
        for j, c in enumerate(NUM):
            try:
                v = float(r.get(c, "?"))
                if np.isfinite(v):
                    X[i, j] = v
            except (ValueError, TypeError):
                pass
        o = (r.get(CAT) or "?").strip()
        cat.append(o)
    return X, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    Xt, ct = load(a.train)
    Xs, cs = load(a.test)

    med = np.array([np.nanmedian(Xt[:, j]) if np.isfinite(Xt[:, j]).any() else 0.0
                    for j in range(Xt.shape[1])])
    mad = np.array([1.4826 * np.nanmedian(np.abs(Xt[:, j] - med[j])) if np.isfinite(Xt[:, j]).any() else 0.0
                    for j in range(Xt.shape[1])])
    sd = np.array([np.nanstd(Xt[:, j]) if np.isfinite(Xt[:, j]).any() else 0.0
                   for j in range(Xt.shape[1])])
    scale = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))

    def imp(X):
        X = X.copy()
        idx = np.where(~np.isfinite(X))
        X[idx] = med[idx[1]]
        return X

    Ti, Si = imp(Xt), imp(Xs)
    mu = Ti.mean(0)
    cov = np.cov(Ti.T) + np.eye(Ti.shape[1]) * 1e-6 * np.maximum(np.diag(np.cov(Ti.T)), 1e-6)
    P = np.linalg.pinv(cov)

    known = [c for c in ct if c != "?"]
    n = max(len(known), 1)
    freq = {}
    for c in known:
        freq[c] = freq.get(c, 0) + 1

    def raw(X, cats):
        d = X - mu
        maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, P, d), 0))
        rz = np.abs((X - med) / scale).max(1)
        pen = np.array([0.0 if c == "?" else (3.0 if freq.get(c, 0) / n < 0.02 else 0.0) for c in cats])
        return maha + rz + pen

    rt = np.sort(raw(Ti, ct))
    rs = raw(Si, cs)
    nt = len(rt)
    scores = np.searchsorted(rt, rs, side="right") / max(nt, 1)
    scores = np.clip(scores, 0.0, 1.0)
    thr = 0.975
    out = {"scores": [float(s) for s in scores],
           "alerts": [bool(s > thr) for s in scores],
           "threshold": thr}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()

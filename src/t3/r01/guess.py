import sys, json, csv, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return [h.strip() for h in rows[0]], rows[1:]


def num(v):
    try:
        v = v.strip()
        return np.nan if v in ("?", "") else float(v)
    except Exception:
        return np.nan


def col(h, rows, name):
    # find column by name (ignoring +/-/X suffix)
    for i, n in enumerate(h):
        if n.rstrip("+-X") == name:
            return [r[i] if i < len(r) else "?" for r in rows]
    return ["?"] * len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    th, tr = load(a.train)
    eh, te = load(a.test)

    def numeric(h, rows):
        return np.array([[num(v) for v in col(h, rows, n)]
                         for n in ("Clndrs", "Volume", "Model")]).T

    Xn_tr, Xn_te = numeric(th, tr), numeric(eh, te)
    o_tr = [v.strip() for v in col(th, tr, "origin")]
    o_te = [v.strip() for v in col(eh, te, "origin")]
    cats = sorted(set(o_tr))
    oh = lambda o: np.array([[x == c for c in cats] for x in o], float).reshape(len(o), len(cats))
    O_tr, O_te = oh(o_tr), oh(o_te)

    # median-fill Clndrs and Model
    for j in (0, 2):
        m = np.nanmedian(Xn_tr[:, j]) if not np.all(np.isnan(Xn_tr[:, j])) else 0
        for X in (Xn_tr, Xn_te):
            X[np.isnan(X[:, j]), j] = m

    # impute Volume from other features
    def other(Xn, O):
        return np.hstack([Xn[:, [0, 2]], O])
    vol_ok = ~np.isnan(Xn_tr[:, 1])
    if vol_ok.sum() > 3:
        vm = Ridge(alpha=1.0).fit(other(Xn_tr, O_tr)[vol_ok], Xn_tr[vol_ok, 1])
        vmed = None
    else:
        vm, vmed = None, 0.0
    for X, O in ((Xn_tr, O_tr), (Xn_te, O_te)):
        miss = np.isnan(X[:, 1])
        if miss.any():
            X[miss, 1] = vm.predict(other(X, O)[miss]) if vm else vmed

    F_tr = np.hstack([Xn_tr, O_tr])
    F_te = np.hstack([Xn_te, O_te])

    lbs = np.array([num(v) for v in col(th, tr, "Lbs")])
    acc = np.array([num(v) for v in col(th, tr, "Acc")])
    mpg = np.array([num(v) for v in col(th, tr, "Mpg")])

    def norm(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    d2h = np.sqrt(((norm(lbs) - 0) ** 2 + (1 - norm(acc)) ** 2 + (1 - norm(mpg)) ** 2) / 3)

    def fit_pred(y):
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            return np.zeros(len(F_te))
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
        m.fit(F_tr[ok], y[ok])
        return m.predict(F_te)

    out = {"mpg": [float(v) for v in fit_pred(mpg)],
           "scores": [float(v) for v in fit_pred(d2h)]}
    print(json.dumps(out))


main()

import sys, csv, json, math, time, argparse
from collections import deque

W = 20
K = 5


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    cols = []
    for h in hdr:
        name = h.rstrip('+-X')
        cols.append(dict(raw=h, name=name, num=h[0].isupper(),
                         ign=h.endswith('X'),
                         goal=1 if h.endswith('+') else (0 if h.endswith('-') else None)))
    data = []
    for r in rows[1:]:
        if not r or all(not c.strip() for c in r):
            continue
        row = []
        for c, v in zip(cols, r):
            v = v.strip()
            if v == '?' or v == '':
                row.append(None)
            elif c['num']:
                try:
                    row.append(float(v))
                except ValueError:
                    row.append(None)
            else:
                row.append(v)
        row += [None] * (len(cols) - len(row))
        data.append(row)
    return cols, data


def ks_p(a, b):
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-9:
        return 1.0
    p = 0.0
    for k in range(1, 101):
        p += 2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
    return min(1.0, max(0.0, p))


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    a = ap.parse_args()
    cols, train = load(a.train)
    _, test = load(a.test)
    nc = len(cols)
    goals = [i for i, c in enumerate(cols) if c['goal'] is not None]
    feats = [i for i in range(nc) if not cols[i]['ign']]
    mpg_i = next((i for i in goals if cols[i]['name'].lower().startswith('mpg')), goals[-1])

    lo, hi = {}, {}
    for i in range(nc):
        if cols[i]['num']:
            v = [r[i] for r in train if r[i] is not None]
            lo[i], hi[i] = (min(v), max(v)) if v else (0, 1)
            if hi[i] == lo[i]:
                hi[i] = lo[i] + 1

    def score(vals):
        s = 0.0
        for i in goals:
            x = (vals[i] - lo[i]) / (hi[i] - lo[i])
            s += (cols[i]['goal'] - x) ** 2
        return math.sqrt(s / len(goals))

    def predict(row, t):
        fs = [i for i in feats if i != t and row[i] is not None]
        cand = []
        for r in train:
            if r[t] is None:
                continue
            d = c = 0
            for i in fs:
                if r[i] is None:
                    continue
                if cols[i]['num']:
                    d += abs(r[i] - row[i]) / (hi[i] - lo[i])
                else:
                    d += 0.0 if r[i] == row[i] else 1.0
                c += 1
            cand.append((d / c if c else 1.0, r[t]))
        cand.sort(key=lambda x: x[0])
        nb = cand[:K]
        if not nb:
            return (lo[t] + hi[t]) / 2
        w = [1 / (d + 1e-3) for d, _ in nb]
        return sum(wi * v for wi, (_, v) in zip(w, nb)) / sum(w)

    hist = []
    for r in train:
        if all(r[i] is not None for i in goals):
            hist.append(score(r))

    mpgs, scores, alerts = [], [], []
    win = deque(maxlen=W)
    for row in test:
        row = list(row) + [None] * (nc - len(row))
        full = list(row)
        pm = None
        for t in goals:
            if row[t] is None:
                full[t] = predict(row, t)
        pm = full[mpg_i]
        s = score(full)
        mpgs.append(pm)
        scores.append(s)
        win.append(s)
        alert = False
        if len(win) == W and hist:
            alert = ks_p(list(win), hist) < 0.05
        alerts.append(bool(alert))

    print(json.dumps({'mpg': mpgs, 'scores': scores, 'alerts': alerts,
                      'time_sec': time.time() - t0}))


if __name__ == '__main__':
    main()

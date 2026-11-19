import sys, csv, json, math, argparse

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def med(a):
    a = sorted(a); n = len(a)
    return 0.0 if n == 0 else (a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2)

def scale(a, m):
    s = 1.4826 * med([abs(x - m) for x in a])
    if s < 1e-9:
        s = math.sqrt(sum((x - m) ** 2 for x in a) / max(1, len(a))) 
    return s if s > 1e-9 else 1.0

def load(path):
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [x for x in r[1:] if x]

def feats(h):
    nc, cc = [], []
    for i, n in enumerate(h):
        if not n or n[-1] in "X+-":
            continue
        (nc if n[0].isupper() else cc).append(i)
    return nc, cc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--train", required=True); p.add_argument("--test", required=True)
    a = p.parse_args()
    th, tr = load(a.train)
    eh, te = load(a.test)
    nc, cc = feats(th)
    # map to test columns by name
    idx = {n: i for i, n in enumerate(eh)}
    nc = [i for i in nc if th[i] in idx]
    cc = [i for i in cc if th[i] in idx]

    def get(row, h, i, isnum):
        j = i if h is th else idx[th[i]]
        if j >= len(row): return None
        v = row[j].strip()
        if v in ("", "?"): return None
        return num(v) if isnum else v

    def col(i):
        return [x for x in (get(r, th, i, True) for r in tr) if x is not None]

    st = {}
    for i in nc:
        c = col(i); m = med(c); st[i] = (m, scale(c, m))
    cats = {}
    for i in cc:
        vals = [get(r, th, i, False) for r in tr]
        vals = [v for v in vals if v is not None]
        d = {}
        for v in vals: d[v] = d.get(v, 0) + 1
        cats[i] = (d, max(1, len(vals)))
    pairs = {}
    for x in nc:
        for y in nc:
            if x == y: continue
            pts = [(get(r, th, x, True), get(r, th, y, True)) for r in tr]
            pts = [q for q in pts if q[0] is not None and q[1] is not None]
            if len(pts) < 5: continue
            mx = sum(q[0] for q in pts) / len(pts); my = sum(q[1] for q in pts) / len(pts)
            sxx = sum((q[0] - mx) ** 2 for q in pts)
            if sxx < 1e-9: continue
            b = sum((q[0] - mx) * (q[1] - my) for q in pts) / sxx
            c0 = my - b * mx
            res = [q[1] - (c0 + b * q[0]) for q in pts]
            mr = med(res)
            pairs[(x, y)] = (b, c0, mr, max(scale(res, mr), 0.05 * st[y][1]))

    def z(row, h):
        zs = [0.0]
        v = {}
        for i in nc:
            v[i] = get(row, h, i, True)
            if v[i] is not None:
                m, s = st[i]; zs.append(abs(v[i] - m) / s)
        for (x, y), (b, c0, mr, s) in pairs.items():
            if v.get(x) is not None and v.get(y) is not None:
                zs.append(abs(v[y] - (c0 + b * v[x]) - mr) / s)
        for i in cc:
            val = get(row, h, i, False)
            if val is None: continue
            d, n = cats[i]
            f = d.get(val, 0) / n
            if f == 0: zs.append(8.0)
            elif f < 0.05: zs.append(3.0 + (0.05 - f) * 40)
        return max(zs)

    trz = sorted(z(r, th) for r in tr)
    q = trz[min(len(trz) - 1, int(0.97 * len(trz)))] if trz else 3.5
    tz = max(3.5, q * 1.1)
    f = lambda s: s / (s + 4.0)
    zs = [z(r, eh) for r in te]
    print(json.dumps({"scores": [f(s) for s in zs], "alerts": [s > tz for s in zs], "threshold": f(tz)}))

main()

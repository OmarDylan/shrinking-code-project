import csv, sys, json, math

def load(p):
    with open(p, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        rows = [x for x in r if x]
    return h, rows

def val(s, num):
    s = s.strip()
    if s == "?" or s == "":
        return None
    return float(s) if num else s

def stats(xs):
    n = len(xs)
    m = sum(xs) / n if n else 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd

def mk(m, sd):
    def norm(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return norm

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    h, R = load(tr)
    h2, T = load(te)
    cols = {}
    for i, n in enumerate(h):
        base = n.rstrip("+-")
        if n.endswith("X"):
            continue
        cols[n] = (i, base, n[0].isupper(), n[-1] if n[-1] in "+-" else "")
    goals = [(n, c) for n, c in cols.items() if c[3]]
    feats = [(n, c) for n, c in cols.items() if not c[3]]
    gnorm = {}
    for n, c in goals:
        xs = [val(r[c[0]], True) for r in R]
        xs = [x for x in xs if x is not None]
        gnorm[n] = mk(*stats(xs))
    heaven = {n: 0.0 if c[3] == "-" else 1.0 for n, c in goals}
    d2h = []
    for r in R:
        s = 0
        for n, c in goals:
            v = val(r[c[0]], True)
            s += (gnorm[n](v) - heaven[n]) ** 2
        d2h.append(math.sqrt(s / len(goals)))
    order = sorted(range(len(R)), key=lambda i: (round(d2h[i], 12), i))
    q = max(1, len(R) // 4)
    good, poor = order[:q], order[-q:]
    fi = {}
    for n, c in feats:
        i = c[0]
        if c[2]:
            xs = [val(r[i], True) for r in R]
            xs = [x for x in xs if x is not None]
            nm = mk(*stats(xs))
            cen = []
            for g in (good, poor):
                vs = [val(R[k][i], True) for k in g]
                vs = [v for v in vs if v is not None]
                cen.append(nm(sum(vs) / len(vs)) if vs else None)
            fi[n] = ("n", i, nm, cen)
        else:
            fr = []
            for g in (good, poor):
                vs = [val(R[k][i], False) for k in g]
                vs = [v for v in vs if v is not None]
                d = {}
                for v in vs:
                    d[v] = d.get(v, 0) + 1
                fr.append((d, len(vs)))
            fi[n] = ("c", i, fr)
    out = []
    for r in T:
        gs = {}
        for n, c in goals:
            v = val(r[c[0]], True)
            gs[n] = abs(gnorm[n](v) - heaven[n]) if v is not None else 0.0
        fs = {}
        for n, f in fi.items():
            if f[0] == "n":
                x = val(r[f[1]], True)
                gc, pc = f[3]
                if x is None or gc is None or pc is None:
                    fs[n] = 0.0
                else:
                    z = f[2](x)
                    fs[n] = max(0.0, abs(z - gc) - abs(z - pc))
            else:
                x = val(r[f[1]], False)
                if x is None:
                    fs[n] = 0.0
                else:
                    (gd, gn), (pd, pn) = f[2]
                    gf = gd.get(x, 0) / gn if gn else 0.0
                    pf = pd.get(x, 0) / pn if pn else 0.0
                    fs[n] = max(0.0, pf - gf)
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": gs, "features": fs,
                    "goal_order": key(gs), "feature_order": key(fs)})
    sys.stdout.write(json.dumps({"diagnoses": out}))

main()

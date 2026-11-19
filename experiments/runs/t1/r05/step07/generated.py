import csv, sys, json, math, argparse

GOALS = ["Lbs-", "Acc+", "Mpg+"]
NUM = ["Clndrs", "Volume", "Model"]

def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def fl(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def ms(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    tr, te = rd(a.train), rd(a.test)

    gst = {g: ms([float(r[g]) for r in tr]) for g in GOALS}

    def d2h(gv):
        s = 0
        for g in GOALS:
            m, sd = gst[g]
            z = max(-3, min(3, (gv[g] - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - (0 if g == "Lbs-" else 1)) ** 2
        return math.sqrt(s / 3)

    ty = [d2h({g: float(r[g]) for g in GOALS}) for r in tr]
    nst = {}
    for c in NUM:
        v = [fl(r[c]) for r in tr]
        v = [x for x in v if x is not None]
        nst[c] = ms(v) if v else (0, 1)

    def feat(r):
        return [(None if fl(r.get(c)) is None else (fl(r[c]) - nst[c][0]) / (nst[c][1] + 1e-9)) for c in NUM]

    trf = [feat(r) for r in tr]

    def pred(r):
        f = feat(r)
        o = r.get("origin")
        ds = []
        for i, t in enumerate(trf):
            s = 0.0
            n = 0
            for x, y in zip(f, t):
                if x is not None and y is not None:
                    s += (x - y) ** 2
                    n += 1
            s = s / n if n else 1.0
            if o not in (None, "?") and tr[i].get("origin") not in (None, "?") and o != tr[i]["origin"]:
                s += 1.0
            ds.append((s, i))
        ds.sort()
        k = ds[:7]
        w = [1 / (d + 0.05) for d, _ in k]
        return sum(wi * ty[i] for wi, (_, i) in zip(w, k)) / sum(w)

    ps = [pred(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (ps[i], i))[:min(10, len(te))]

    obs = {}
    for i in order:
        print(json.dumps({"inspect": i}), flush=True)
        line = sys.stdin.readline()
        resp = json.loads(line)
        obs[i] = d2h({g: float(resp["goals"][g]) for g in GOALS})
    rk = sorted(obs, key=lambda i: (round(obs[i], 12), i))[:3]
    print(json.dumps({"selected": rk}), flush=True)

main()

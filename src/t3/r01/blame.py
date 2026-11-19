import sys, csv, json, math, argparse

def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    hdr, tr = load(a.train)
    h2, te = load(a.test)
    cols = []
    for i, h in enumerate(hdr):
        if h.endswith("X"):
            continue
        num = h[0].isupper()
        goal = h.endswith("+") or h.endswith("-")
        cols.append((i, h, num, goal, 1.0 if h.endswith("+") else 0.0))
    idx2 = {h: i for i, h in enumerate(h2)}

    def val(row, i, h, num):
        j = idx2.get(h, i)
        if j >= len(row):
            return None
        s = row[j]
        if s == "?" or s == "":
            return None
        if num:
            try:
                return float(s)
            except ValueError:
                return None
        return s

    def parse(rows):
        return [{h: val(r, i, h, n) for i, h, n, g, _ in cols} for r in rows]
    T, E = parse(tr), parse(te)

    rng = {}
    for i, h, n, g, hv in cols:
        if n:
            v = [r[h] for r in T if r[h] is not None]
            rng[h] = (min(v), max(v)) if v else (0.0, 1.0)

    def norm(h, x):
        lo, hi = rng[h]
        return (x - lo) / (hi - lo + 1e-32)

    goals = [c for c in cols if c[3]]
    inputs = [c for c in cols if not c[3]]

    def gaps(r):
        return {h: abs(norm(h, r[h]) - hv) for _, h, n, g, hv in goals if r[h] is not None}

    scored = []
    for r in T:
        gp = gaps(r)
        if len(gp) == len(goals) and goals:
            scored.append((math.sqrt(sum(x * x for x in gp.values()) / len(gp)), r))
    scored.sort(key=lambda t: t[0])
    k = max(1, len(scored) // 4)
    good = [r for _, r in scored[:k]]
    poor = [r for _, r in scored[-k:]]

    prof = {}
    for _, h, n, g, _ in inputs:
        if n:
            def center(grp):
                v = [norm(h, r[h]) for r in grp if r[h] is not None]
                return sum(v) / len(v) if v else None
            prof[h] = (center(good), center(poor))
        else:
            def freq(grp):
                d = {}
                for r in grp:
                    if r[h] is not None:
                        d[r[h]] = d.get(r[h], 0) + 1
                return {x: c / len(grp) for x, c in d.items()}
            prof[h] = (freq(good), freq(poor))

    out = []
    for r in E:
        gp = gaps(r)
        if not gp:
            continue
        feats = {}
        for _, h, n, g, _ in inputs:
            x = r[h]
            s = 0.0
            if x is not None:
                if n:
                    gc, pc = prof[h]
                    if gc is not None and pc is not None:
                        xv = norm(h, x)
                        s = max(0.0, abs(xv - gc) - abs(xv - pc))
                else:
                    gf, pf = prof[h]
                    s = max(0.0, pf.get(x, 0.0) - gf.get(x, 0.0))
            feats[h] = s
        out.append({
            "goals": gp,
            "features": feats,
            "goal_order": sorted(gp, key=lambda z: (-gp[z], z)),
            "feature_order": sorted(feats, key=lambda z: (-feats[z], z)),
        })
    print(json.dumps({"diagnoses": out}))

main()

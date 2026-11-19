import sys, json, csv, math

def mean(a): return sum(a) / len(a)
def var(a):
    m = mean(a); return sum((x - m) ** 2 for x in a) / (len(a) - 1)
def sign(x): return (x > 0) - (x < 0)

def certify(xs, ys):
    n, m = len(xs), len(ys)
    pv = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    c = abs(mean(xs) - mean(ys)) / (math.sqrt(pv) + 1e-32)
    cl = abs(sum(sign(x - y) for x in xs for y in ys)) / (n * m)
    ks = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
             for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
    return {"cohen": c, "cliffs": cl, "ks": ks, "same": c <= 0.35 and cl <= 0.197 and ks <= 1.36}

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def main():
    a = sys.argv
    train_p = a[a.index("--train") + 1]
    test_p = a[a.index("--test") + 1]
    with open(train_p, newline="") as f:
        rd = list(csv.reader(f))
    H = [h.strip() for h in rd[0]]
    tr = [r for r in rd[1:] if r]
    with open(test_p) as f:
        case = json.load(f)
    ix = {h: i for i, h in enumerate(H)}
    goals = ["Lbs-", "Acc+", "Mpg+"]
    heaven = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
    st = {}
    for g in goals:
        v = [x for x in (num(r[ix[g]]) for r in tr) if x is not None]
        st[g] = (mean(v), math.sqrt(var(v)))
    def nm(v, m, s):
        z = max(-3, min(3, (v - m) / (s + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    def d2h(gd):
        return math.sqrt(sum((nm(gd[g], *st[g]) - heaven[g]) ** 2 for g in goals) / 3)
    tD = [d2h({g: num(r[ix[g]]) for g in goals}) for r in tr]
    target = sorted(tD)[len(tr) // 4]
    nf = ["Clndrs", "Volume", "Model"]
    fs = {}
    for g in nf:
        v = [x for x in (num(r[ix[g]]) for r in tr) if x is not None]
        fs[g] = (mean(v), math.sqrt(var(v)))
    def feat(r):
        out = []
        for g in nf:
            v = num(r[ix[g]])
            out.append(None if v is None else nm(v, *fs[g]))
        o = r[ix["origin"]]
        out.append(None if o in ("?", None, "") else o)
        return out
    def dist(p, q):
        gp = []
        for k in range(4):
            if p[k] is None or q[k] is None: continue
            gp.append((p[k] - q[k]) ** 2 if k < 3 else (0 if p[k] == q[k] else 1))
        return math.sqrt(sum(gp) / len(gp)) if gp else 1.0
    trF = [feat(r) for r in tr]
    rows = case["rows"]
    teF = {}
    def tf(i):
        if i not in teF: teF[i] = feat(rows[i])
        return teF[i]
    predc = {}
    def pred(i):
        if i not in predc:
            f = tf(i)
            ds = sorted((round(dist(f, t), 12), j) for j, t in enumerate(trF))[:5]
            predc[i] = mean([tD[j] for _, j in ds])
        return predc[i]

    def ask(e, s, i):
        print(json.dumps({"episode": e, "strategy": s, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h({g: float(resp["goals"][g]) for g in goals})

    def metrics(seq):
        lt = 11
        for k, d in enumerate(seq):
            if round(d, 12) <= round(target, 12):
                lt = k + 1; break
        return {"best5": min(seq[:5]), "best10": min(seq), "labels_to_target": lt}

    results = []
    for e, ep in enumerate(case["episodes"]):
        pool = list(ep["pool"])
        obs = []
        left = list(pool)
        seq = []
        for _ in range(10):
            def sc(i):
                num_, den = 0.0, 1.0
                for j, d in obs:
                    w = math.exp(-5 * dist(tf(i), tf(j)))
                    num_ += w * (d - pred(j)); den += w
                return (pred(i) + num_ / den, i)
            i = min(left, key=sc)
            left.remove(i)
            d = ask(e, "adaptive", i)
            obs.append((i, d)); seq.append(d)
        ad = metrics(seq)
        seq = [ask(e, "random", i) for i in ep["random_order"][:10]]
        results.append({"episode": e, "adaptive": ad, "random": metrics(seq)})
    comp = {}
    for k in ("best5", "best10", "labels_to_target"):
        comp[k] = certify([float(r["adaptive"][k]) for r in results],
                          [float(r["random"][k]) for r in results])
    print(json.dumps({"results": results, "comparisons": comp}), flush=True)

main()

import sys, csv, json, math

def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hn, train = load(tr)
    tn, test = load(te)
    G = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
    gi = {g: hn.index(g) for g in G}
    NX = ["Clndrs", "Volume", "Model"]

    # goal stats
    st = {}
    for g in G:
        v = [x for x in (num(r[gi[g]]) for r in train) if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        st[g] = (m, sd)

    def d2h(goals):
        s = 0
        for g, h in G.items():
            m, sd = st[g]
            z = max(-3, min(3, (goals[g] - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(s / 3)

    # training labelled rows
    tl = []
    for r in train:
        gs = {g: num(r[gi[g]]) for g in G}
        if all(v is not None for v in gs.values()):
            tl.append((r, d2h(gs)))

    # x normalization
    xs = {}
    for c in NX:
        v = [x for x in (num(r[hn.index(c)]) for r in train) if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1
        xs[c] = (m, sd)

    def vec(names, r):
        out = []
        for c in NX:
            x = num(r[names.index(c)])
            out.append(None if x is None else (x - xs[c][0]) / xs[c][1])
        o = r[names.index("origin")]
        out.append(None if o == "?" else o)
        return out

    def dist(p, q):
        s = 0
        for x, y in zip(p, q):
            if x is None or y is None:
                s += 1
            elif isinstance(x, str):
                s += x != y
            else:
                s += min(abs(x - y), 3) ** 2 / 4
        return s

    tv = [(vec(hn, r), d) for r, d in tl]

    def pred(r):
        v = vec(tn, r)
        ds = sorted((dist(v, p), d) for p, d in tv)[:5]
        return sum(d for _, d in ds) / len(ds) if ds else 0.5

    preds = sorted((pred(r), i) for i, r in enumerate(test))
    picks = [i for _, i in preds[:10]]

    seen = {}
    for i in picks:
        print(json.dumps({"inspect": i}))
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        seen[i] = d2h(resp["goals"])

    top = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    print(json.dumps({"selected": top}))
    sys.stdout.flush()

main()

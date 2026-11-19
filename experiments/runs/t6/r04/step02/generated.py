import csv, sys, json, argparse
from math import exp, sqrt

def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def get(r, i):
    return r[i] if i < len(r) else "?"

def goal_stats(names, tr):
    goals = [(i, 0 if n[-1] == "-" else 1) for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        v = [float(r[i]) for r in tr if get(r, i) != "?"]
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (mu, sd)
    return goals, stats

def make_d2h(goals, stats):
    def nrm(i, x):
        mu, sd = stats[i]
        z = max(-3.0, min(3.0, (x - mu) / (sd + 1e-32)))
        return 1 / (1 + exp(-1.702 * z))
    return lambda vals: sqrt(sum((nrm(i, vals[i]) - h) ** 2 for i, h in goals) / len(goals))

def remember(names, tr, te):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    scores = [d2h({i: float(r[i]) for i, _ in goals}) for r in te]
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

def guess(names, tr, te, k=7):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    gi = {i for i, _ in goals}
    mpg_i = next(i for i, n in enumerate(names) if n.startswith("Mpg"))
    xs = [i for i, n in enumerate(names) if n[-1] not in "+-X" and i not in gi]
    tr = [r for r in tr if all(get(r, i) != "?" for i in gi)]
    sds = {}
    for i in xs:
        if names[i][0].isupper():
            v = [float(r[i]) for r in tr if get(r, i) != "?"]
            mu = sum(v) / len(v)
            sds[i] = sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
    ty = [(float(r[mpg_i]), d2h({i: float(r[i]) for i in gi})) for r in tr]
    mm = sum(y[0] for y in ty) / len(ty)
    ms = sum(y[1] for y in ty) / len(ty)

    def dist(a, b):
        s = n = 0
        for i in xs:
            p, q = get(a, i), get(b, i)
            if p == "?" or q == "?":
                continue
            s += ((float(p) - float(q)) / sds[i]) ** 2 if i in sds else float(p != q)
            n += 1
        return sqrt(s / n) if n else 1.0

    mpg, scores = [], []
    for r in te:
        near = sorted(range(len(tr)), key=lambda j: dist(r, tr[j]))[:k]
        w = [1 / (dist(r, tr[j]) + 0.05) for j in near]
        sw = sum(w)
        m = sum(wi * ty[j][0] for wi, j in zip(w, near)) / sw
        s = sum(wi * ty[j][1] for wi, j in zip(w, near)) / sw
        mpg.append(m if m > 0 else mm)
        scores.append(min(1.0, max(0.0, s)) if s == s else ms)
    return {"mpg": mpg, "scores": scores}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd not in ("remember", "guess"):
        sys.exit(1)
    names, tr = load(a.train)
    _, te = load(a.test)
    out = remember(names, tr, te) if a.cmd == "remember" else guess(names, tr, te)
    print(json.dumps(out))

main()

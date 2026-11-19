import sys, json, time, argparse
from math import exp, sqrt
from bisect import bisect_right
from statistics import median


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f if l.strip()]
    names = [s.strip() for s in lines[0].split(",")]
    rows = []
    for l in lines[1:]:
        r = []
        for s in l.split(","):
            s = s.strip()
            try:
                r.append(int(s))
            except ValueError:
                try:
                    r.append(float(s))
                except ValueError:
                    r.append(s)
        rows.append(r)
    return names, rows


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    names, train = load(a.train)
    _, test = load(a.test)
    use = [i for i, s in enumerate(names) if s[-1] != "X"]
    ys = [i for i in use if names[i][-1] in "+-"]
    xs = [i for i in use if i not in ys]
    goal = {i: (0 if names[i][-1] == "-" else 1) for i in ys}
    st = {}
    for i in use:
        if names[i][0].isupper():
            v = [r[i] for r in train if r[i] != "?"]
            mu = sum(v) / len(v)
            sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
            st[i] = (mu, sd)

    def gaps(r):
        g = {}
        for i in ys:
            v = r[i]
            if v == "?":
                g[i] = 1.0
                continue
            mu, sd = st[i]
            z = max(-3, min(3, (v - mu) / (1e-32 + sd)))
            g[i] = abs(1 / (1 + exp(-1.7 * z)) - goal[i])
        return g

    def score(r):
        g = gaps(r)
        return sqrt(sum(x * x for x in g.values()) / len(ys))

    tr = [score(r) for r in train]
    cutoff = median(tr)
    overall = sum(tr) / len(tr)
    edges = {}
    for i in xs:
        if i in st:
            v = sorted(r[i] for r in train if r[i] != "?")
            edges[i] = [v[len(v) * k // 4] for k in (1, 2, 3)]

    def key(i, v):
        if v == "?":
            return None
        return bisect_right(edges[i], v) if i in edges else v

    groups = {}
    for r, s in zip(train, tr):
        for i in xs:
            k = key(i, r[i])
            if k is not None:
                groups.setdefault((i, k), []).append(s)

    def feature(r, acc):
        best, bl = None, 0.0
        for i in xs:
            k = key(i, r[i])
            g = groups.get((i, k)) if k is not None else None
            if not g:
                continue
            lift = overall - sum(g) / len(g)
            if not acc:
                lift = -lift
            if lift > bl + 1e-12:
                best, bl = names[i], lift
        return best

    n = len(test)
    out = []
    for idx in range(n):
        j = (idx + 1) % n
        r, o = test[idx], test[j]
        s, so = score(r), score(o)
        acc = s <= cutoff
        gr, go = gaps(r), gaps(o)
        weak = names[max(ys, key=lambda i: gr[i])]
        if round(s, 12) == round(so, 12):
            win = adv = None
        else:
            win = idx if s < so else j
            wg, lg = (gr, go) if win == idx else (go, gr)
            adv = names[max(ys, key=lambda i: lg[i] ** 2 - wg[i] ** 2)]
        feat = feature(r, acc)
        if win is None:
            cmp = f"Car {idx} and car {j} tie (scores {s:.6f} vs {so:.6f})."
        else:
            lose = j if win == idx else idx
            cmp = (f"Car {win} is recommended over car {lose} (lower distance to heaven "
                   f"{min(s, so):.4f} vs {max(s, so):.4f}); the biggest advantage is {adv}.")
        text = (f"{cmp} Car {idx} is {'accepted' if acc else 'rejected'}: score {s:.4f} "
                f"{'<=' if acc else '>'} median training cutoff {cutoff:.4f}; "
                f"its weakest goal is {weak}.")
        if feat:
            text += f" Feature {feat} is associated with this outcome (retrospective, not causal)."
        out.append({"row": idx, "other": j, "score": s, "other_score": so,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": feat, "text": text})
    print(json.dumps({"explanations": out, "count": n,
                      "execution_time": time.time() - t0}))


if __name__ == "__main__":
    main()

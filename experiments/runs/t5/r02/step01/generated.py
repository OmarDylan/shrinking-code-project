from math import exp, sqrt
import sys, json

def read(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in l.split(",")] for l in f if l.strip()]
    return rows[0], rows[1:]

def main():
    a = sys.argv[1:]
    if not a or a[0] != "remember":
        sys.exit("usage: main.py remember --train TRAIN --test TEST")
    train, test = a[a.index("--train") + 1], a[a.index("--test") + 1]
    names, tr = read(train)
    _, te = read(test)
    goals = [(i, n[-1] == "+") for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        v = [float(r[i]) for r in tr]
        m = sum(v) / len(v)
        sd = sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (m, sd)

    def score(r):
        s = 0.0
        for i, h in goals:
            m, sd = stats[i]
            z = max(-3.0, min(3.0, (float(r[i]) - m) / (sd + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - (1 if h else 0)) ** 2
        return sqrt(s / len(goals))

    sc = [score(r) for r in te]
    rank = sorted(range(len(sc)), key=lambda i: (sc[i], i))
    print(json.dumps({"scores": sc, "top": rank[:10], "best": rank[0],
                      "median": rank[len(rank) // 2], "worst": rank[-1]}))

main()

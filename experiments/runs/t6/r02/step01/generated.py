import sys, csv, json, argparse
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}

def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    head = rows[0]
    ix = [head.index(g) for g in GOALS]
    data = [[float(r[i]) for i in ix] for r in rows[1:]
            if all(r[i] != "?" for i in ix)]
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd != "remember":
        sys.exit("unknown command")
    tr, te = load(a.train), load(a.test)
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        mu = sum(col) / n
        sd = sqrt(sum((v - mu) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((mu, sd))
    hs = list(GOALS.values())

    def score(r):
        s = 0.0
        for j in range(3):
            mu, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - mu) / (sd + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - hs[j]) ** 2
        return sqrt(s / 3)

    scores = [score(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    out = {"scores": scores, "top": order[:10], "best": order[0],
           "median": order[len(te) // 2], "worst": order[-1]}
    sys.stdout.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    main()

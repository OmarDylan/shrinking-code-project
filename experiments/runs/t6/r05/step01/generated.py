import sys, csv, json, argparse
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}

def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    names, tr = load(a.train)
    tnames, te = load(a.test)
    stats = {}
    for g, h in GOALS.items():
        i = names.index(g)
        xs = [float(r[i]) for r in tr]
        n = len(xs)
        mu = sum(xs) / n
        sd = sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats[g] = (tnames.index(g), mu, sd, h)
    scores = []
    for r in te:
        s = 0.0
        for i, mu, sd, h in stats.values():
            z = max(-3.0, min(3.0, (float(r[i]) - mu) / (sd + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - h) ** 2
        scores.append(sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    n = len(order)
    print(json.dumps({"scores": scores, "top": order[:10],
                      "best": order[0], "median": order[n // 2],
                      "worst": order[-1]}))

main()

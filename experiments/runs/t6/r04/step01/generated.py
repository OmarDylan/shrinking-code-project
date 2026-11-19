import csv, sys, json, argparse
from math import exp, sqrt

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
    if a.cmd != "remember":
        sys.exit(1)
    names, tr = load(a.train)
    _, te = load(a.test)
    goals = [(i, 0 if n[-1] == "-" else 1) for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        v = [float(r[i]) for r in tr if r[i] != "?"]
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (mu, sd)

    def nrm(i, x):
        mu, sd = stats[i]
        z = max(-3.0, min(3.0, (x - mu) / (sd + 1e-32)))
        return 1 / (1 + exp(-1.702 * z))

    scores = [sqrt(sum((nrm(i, float(r[i])) - h) ** 2 for i, h in goals) / len(goals))
              for r in te]
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    print(json.dumps({"scores": scores, "top": order[:10], "best": order[0],
                      "median": order[len(order) // 2], "worst": order[-1]}))

main()

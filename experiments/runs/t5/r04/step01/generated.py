import sys, csv, json, argparse
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
        sys.exit("unknown command")
    hdr, tr = load(a.train)
    _, te = load(a.test)
    goals = [(i, 0 if h.endswith("-") else 1)
             for i, h in enumerate(hdr) if h.endswith(("-", "+"))]
    stats = {}
    for i, _ in goals:
        v = [float(r[i]) for r in tr if r[i] != "?"]
        n = len(v)
        mu = sum(v) / n
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[i] = (mu, sd)

    def score(r):
        s = 0.0
        for i, h in goals:
            mu, sd = stats[i]
            z = max(-3.0, min(3.0, (float(r[i]) - mu) / (sd + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - h) ** 2
        return sqrt(s / len(goals))

    scores = [score(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    print(json.dumps({"scores": scores, "top": order[:10], "best": order[0],
                      "median": order[len(te) // 2], "worst": order[-1]}))

main()

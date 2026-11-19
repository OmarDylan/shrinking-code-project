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
    names, tr = load(a.train)
    names2, te = load(a.test)
    goals = [i for i, s in enumerate(names) if s[-1] in "+-"]
    stats = {}
    for i in goals:
        vs = [float(r[i]) for r in tr if r[i] != "?"]
        n = len(vs)
        mu = sum(vs) / n
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (n - 1)) if n > 1 else 0.0
        stats[i] = (mu, sd)

    def score(r):
        t = 0.0
        for i in goals:
            mu, sd = stats[i]
            z = max(-3.0, min(3.0, (float(r[i]) - mu) / (sd + 1e-32)))
            nv = 1 / (1 + exp(-1.702 * z))
            h = 0.0 if names[i][-1] == "-" else 1.0
            t += (nv - h) ** 2
        return sqrt(t / len(goals))

    scores = [score(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(te) // 2],
        "worst": order[-1],
    }))


if __name__ == "__main__":
    main()

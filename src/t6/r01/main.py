import sys, json, argparse
from math import exp, sqrt

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f if l.strip()]
    names = [s.strip() for s in lines[0].split(",")]
    rows = [[s.strip() for s in l.split(",")] for l in lines[1:]]
    return names, rows

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    names, train = load(a.train)
    tnames, test = load(a.test)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = {}
    for n, _ in goals:
        i = names.index(n)
        vs = [v for v in (num(r[i]) for r in train if i < len(r)) if v is not None]
        k = len(vs)
        mu = sum(vs) / k if k else 0.0
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (k - 1)) if k > 1 else 0.0
        stats[n] = (mu, sd)

    def d2h(r):
        tot = 0.0
        for n, h in goals:
            i = tnames.index(n)
            v = num(r[i]) if i < len(r) else None
            if v is None:
                tot += 1.0
                continue
            mu, sd = stats[n]
            z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
            x = 1 / (1 + exp(-1.702 * z))
            tot += (x - h) ** 2
        return sqrt(tot / len(goals))

    scores = [d2h(r) for r in test]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))

if __name__ == "__main__":
    main()

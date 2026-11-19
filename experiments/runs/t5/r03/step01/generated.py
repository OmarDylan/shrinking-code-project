import sys, csv, json, argparse
from math import exp, sqrt


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]


def remember(train, test):
    names, tr = load(train)
    tnames, te = load(test)
    goals = [i for i, s in enumerate(names) if s[-1] in "+-"]
    stats = {}
    for g in goals:
        v = [float(r[g]) for r in tr]
        n = len(v)
        mu = sum(v) / n
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[g] = (mu, sd, 0.0 if names[g][-1] == "-" else 1.0)
    scores = []
    for r in te:
        tot = 0.0
        for g in goals:
            mu, sd, h = stats[g]
            z = max(-3.0, min(3.0, (float(r[g]) - mu) / (sd + 1e-32)))
            nv = 1 / (1 + exp(-1.702 * z))
            tot += (nv - h) ** 2
        scores.append(sqrt(tot / len(goals)))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["remember"])
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    sys.stdout.write(json.dumps(remember(a.train, a.test)))


if __name__ == "__main__":
    main()

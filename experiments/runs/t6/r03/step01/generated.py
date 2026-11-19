import sys, json, csv, argparse
from math import exp, sqrt


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    return rows[0], rows[1:]


def remember(train, test):
    names, tr = load(train)
    names2, te = load(test)
    goals = [(i, n[-1]) for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        vs = [float(r[i]) for r in tr if r[i] != "?"]
        mu = sum(vs) / len(vs)
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0.0
        stats[i] = (mu, sd)

    def score(r):
        s = 0.0
        for i, sign in goals:
            mu, sd = stats[i]
            z = max(-3.0, min(3.0, (float(r[i]) - mu) / (sd + 1e-32)))
            nv = 1 / (1 + exp(-1.702 * z))
            h = 1.0 if sign == "+" else 0.0
            s += (nv - h) ** 2
        return sqrt(s / len(goals))

    scores = [score(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(te) // 2], "worst": order[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["remember"])
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    sys.stdout.write(json.dumps(remember(a.train, a.test)))


if __name__ == "__main__":
    main()

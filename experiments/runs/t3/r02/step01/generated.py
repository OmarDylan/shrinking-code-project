import csv, sys, json, math, argparse

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    train, test = load(a.train), load(a.test)
    stats = {}
    for g, _ in GOALS:
        vals = [float(r[g]) for r in train]
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)
    scores = []
    for r in test:
        s = 0.0
        for g, h in GOALS:
            m, sd = stats[g]
            z = max(-3.0, min(3.0, (float(r[g]) - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

main()

import csv, sys, math, json, argparse

G = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}


def load(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def goals(rows):
    out = []
    for row in rows:
        try:
            out.append([float(row[k]) for k in G])
        except (ValueError, TypeError):
            out.append(None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    tr = [g for g in goals(load(a.train)) if g is not None]
    te = goals(load(a.test))
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    h = list(G.values())

    def score(g):
        s = 0.0
        for j in range(3):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (g[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h[j]) ** 2
        return math.sqrt(s / 3)

    scores = [score(g) for g in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))


main()

import csv, sys, json, math, argparse

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    h, tr = load(a.train)
    h2, te = load(a.test)
    goals = [(i, 1.0 if n.endswith("+") else 0.0)
             for i, n in enumerate(h) if n.endswith("+") or n.endswith("-")]
    idx2 = {n: i for i, n in enumerate(h2)}

    stats = {}
    for i, _ in goals:
        v = [x for x in (num(r[i]) if i < len(r) else None for r in tr) if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (m, sd)

    def norm(x, m, sd):
        z = (x - m) / (sd + 1e-32)
        z = max(-3.0, min(3.0, z))
        return 1 / (1 + math.exp(-1.702 * z))

    scores = []
    for r in te:
        s = 0.0
        for i, hv in goals:
            j = idx2.get(h[i], i)
            x = num(r[j]) if j < len(r) else None
            if x is None:
                continue
            s += (norm(x, *stats[i]) - hv) ** 2
        scores.append(math.sqrt(s / len(goals)))

    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))

main()

import sys, csv, json, math, argparse

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}


def load(path):
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]


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

    hdr, tr = load(a.train)
    hdr2, te = load(a.test)
    idx1 = {h: i for i, h in enumerate(hdr)}
    idx2 = {h: i for i, h in enumerate(hdr2)}

    stats = {}
    for g in GOALS:
        vals = [v for r in tr if idx1[g] < len(r) for v in [num(r[idx1[g]])] if v is not None]
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)

    scores = []
    for r in te:
        s = 0.0
        for g, h in GOALS.items():
            v = num(r[idx2[g]]) if idx2[g] < len(r) else None
            if v is None:
                s += 1.0  # missing goal: worst-case penalty
                continue
            m, sd = stats[g]
            z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
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
    print(json.dumps(out))


main()

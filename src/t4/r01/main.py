import csv, sys, json, math, argparse

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    head = [h.strip() for h in rows[0]]
    return head, [[c.strip() for c in r] for r in rows[1:]]


def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def column(head, rows, name):
    i = head.index(name)
    return [num(r[i]) if i < len(r) else None for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    th, tr = load(a.train)
    eh, te = load(a.test)

    stats = {}
    for g in GOALS:
        v = [x for x in column(th, tr, g) if x is not None]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)

    cols = {g: column(eh, te, g) for g in GOALS}

    def norm(x, g):
        if x is None:
            return 0.5
        m, sd = stats[g]
        z = max(-3.0, min(3.0, (x - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    scores = []
    for i in range(len(te)):
        s = sum((norm(cols[g][i], g) - h) ** 2 for g, h in GOALS.items())
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

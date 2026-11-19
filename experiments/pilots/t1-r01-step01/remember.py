import csv, json, sys, math

def load(path):
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]

def num(v):
    try:
        return None if v == "?" else float(v)
    except ValueError:
        return None

def main():
    a = sys.argv[1:]
    if not a or a[0] != "remember":
        sys.exit("usage: main.py remember --train TRAIN --test TEST")
    tr = te = None
    for i, x in enumerate(a):
        if x == "--train": tr = a[i + 1]
        if x == "--test": te = a[i + 1]
    hdr, train = load(tr)
    hdr2, test = load(te)
    goals = [(i, 1.0 if h.endswith("+") else 0.0)
             for i, h in enumerate(hdr) if h.endswith("+") or h.endswith("-")]
    stats = {}
    for i, _ in goals:
        vals = [v for r in train if i < len(r) for v in [num(r[i])] if v is not None]
        stats[i] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    def d2h(r):
        s = n = 0
        for i, heaven in goals:
            v = num(r[i]) if i < len(r) else None
            if v is None:
                continue
            lo, hi = stats[i]
            x = (v - lo) / (hi - lo + 1e-32)
            x = max(0.0, min(1.0, x))
            s += (heaven - x) ** 2
            n += 1
        return math.sqrt(s / n) if n else 1.0

    scores = [d2h(r) for r in test]
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

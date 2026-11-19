import sys, json, argparse, math
from bisect import bisect_left, bisect_right

def cohen(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / m
    vx = sum((a - mx) ** 2 for a in x) / (n - 1) if n > 1 else 0.0
    vy = sum((a - my) ** 2 for a in y) / (m - 1) if m > 1 else 0.0
    df = n + m - 2
    sp = math.sqrt(((n - 1) * vx + (m - 1) * vy) / df) if df > 0 else 0.0
    d = abs(mx - my)
    if sp == 0:
        return 0.0 if d == 0 else 1e9
    return d / sp

def cliffs(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    ys = sorted(y)
    gt = lt = 0
    for a in x:
        lo = bisect_left(ys, a)
        hi = bisect_right(ys, a)
        gt += lo          # y values less than a
        lt += m - hi      # y values greater than a
    return abs(gt - lt) / (n * m)

def ks(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    xs, ys = sorted(x), sorted(y)
    d = 0.0
    for v in set(xs) | set(ys):
        d = max(d, abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m))
    return d / math.sqrt((n + m) / (n * m))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", default=None)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    with open(a.test) as f:
        cases = json.load(f)
    out = []
    for c in cases:
        x = [float(v) for v in c["xs"]]
        y = [float(v) for v in c["ys"]]
        ch, cl, k = cohen(x, y), cliffs(x, y), ks(x, y)
        out.append({"cohen": ch, "cliffs": cl, "ks": k,
                    "same": bool(ch <= 0.35 and cl <= 0.197 and k <= 1.36)})
    print(json.dumps({"comparisons": out}))

if __name__ == "__main__":
    main()

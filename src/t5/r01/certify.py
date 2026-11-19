import sys, json, argparse
from math import sqrt
from bisect import bisect_left, bisect_right

def cohen(xs, ys):
    n, m = len(xs), len(ys)
    a, b = sum(xs) / n, sum(ys) / m
    va = sum((x - a) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    vb = sum((y - b) ** 2 for y in ys) / (m - 1) if m > 1 else 0.0
    df = n + m - 2
    pool = ((n - 1) * va + (m - 1) * vb) / df if df > 0 else 0.0
    d = abs(a - b)
    if pool <= 0:
        return 0.0 if d == 0 else 1e9
    return d / sqrt(pool)

def cliffs(xs, ys):
    ys = sorted(ys)
    m = len(ys)
    gt = lt = 0
    for x in xs:
        lo, hi = bisect_left(ys, x), bisect_right(ys, x)
        gt += lo
        lt += m - hi
    return abs(gt - lt) / (len(xs) * m)

def ks(xs, ys):
    xs, ys = sorted(xs), sorted(ys)
    n, m = len(xs), len(ys)
    d = 0.0
    for v in set(xs) | set(ys):
        d = max(d, abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m))
    return d / sqrt((n + m) / (n * m))

def compare(c):
    xs = [float(v) for v in c["xs"]]
    ys = [float(v) for v in c["ys"]]
    if not xs or not ys:
        return {"cohen": 0.0, "cliffs": 0.0, "ks": 0.0, "same": True}
    co, cl, k = cohen(xs, ys), cliffs(xs, ys), ks(xs, ys)
    return {"cohen": co, "cliffs": cl, "ks": k,
            "same": bool(co <= 0.35 and cl <= 0.197 and k <= 1.36)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", default=None)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    with open(a.test) as f:
        cases = json.load(f)
    if isinstance(cases, dict):
        cases = cases.get("cases", [])
    print(json.dumps({"comparisons": [compare(c) for c in cases]}))

if __name__ == "__main__":
    main()

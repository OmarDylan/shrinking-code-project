import argparse
import json
import math
from bisect import bisect_left, bisect_right


def mean_var(a):
    n = len(a)
    m = math.fsum(a) / n
    v = math.fsum((x - m) ** 2 for x in a) / (n - 1) if n > 1 else 0.0
    return m, v


def cohen(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    m1, v1 = mean_var(xs)
    m2, v2 = mean_var(ys)
    df = n + m - 2
    sd = math.sqrt(((n - 1) * v1 + (m - 1) * v2) / df) if df > 0 else 0.0
    diff = abs(m1 - m2)
    if sd == 0:
        return 0.0 if diff == 0 else 1e12
    return diff / sd


def cliffs(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    s = sorted(ys)
    gt = lt = 0
    for x in xs:
        gt += bisect_left(s, x)          # ys < x
        lt += m - bisect_right(s, x)     # ys > x
    return abs(gt - lt) / (n * m)


def ks(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    a, b = sorted(xs), sorted(ys)
    d = 0.0
    for v in set(a) | set(b):
        f = bisect_right(a, v) / n
        g = bisect_right(b, v) / m
        d = max(d, abs(f - g))
    return d * math.sqrt(n * m / (n + m))


def compare(case):
    xs = [float(x) for x in case["xs"]]
    ys = [float(y) for y in case["ys"]]
    c, l, k = cohen(xs, ys), cliffs(xs, ys), ks(xs, ys)
    same = c <= 0.35 and l <= 0.197 and k <= 1.36
    return {"cohen": c, "cliffs": l, "ks": k, "same": bool(same)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", default=None)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    with open(a.test) as f:
        cases = json.load(f)
    if isinstance(cases, dict):
        cases = cases.get("cases", cases.get("comparisons", []))
    print(json.dumps({"comparisons": [compare(c) for c in cases]}))


if __name__ == "__main__":
    main()

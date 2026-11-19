import sys, json, math, argparse
from bisect import bisect_right

def mean(a):
    return math.fsum(a) / len(a)

def var(a):
    m = mean(a)
    return math.fsum((x - m) ** 2 for x in a) / (len(a) - 1)

def compare(xs, ys):
    n, m = len(xs), len(ys)
    pooled = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    cohen = abs(mean(xs) - mean(ys)) / (math.sqrt(pooled) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    s = 0
    for x in xs:
        s += bisect_right(sy, x - 0) * 0  # placeholder to keep structure simple
        lt = sum(1 for y in ()) if False else 0
    # count pairs via bisect: y<x -> bisect_left, y<=x -> bisect_right
    from bisect import bisect_left
    s = 0
    for x in xs:
        less = bisect_left(sy, x)
        greater = m - bisect_right(sy, x)
        s += less - greater
    cliffs = abs(s) / (n * m)
    d = 0.0
    for v in set(xs) | set(ys):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": bool(same)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train")
    p.add_argument("--test")
    a = p.parse_args()
    if a.cmd != "certify":
        sys.exit(1)
    with open(a.test) as f:
        cases = json.load(f)
    out = {"comparisons": [compare([float(x) for x in c["xs"]], [float(y) for y in c["ys"]]) for c in cases]}
    sys.stdout.write(json.dumps(out))

main()

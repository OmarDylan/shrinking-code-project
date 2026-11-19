import sys, json, math
from bisect import bisect_right

def stats(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs)/n, sum(ys)/m
    vx = sum((x-mx)**2 for x in xs)/(n-1)
    vy = sum((y-my)**2 for y in ys)/(m-1)
    pv = ((n-1)*vx + (m-1)*vy)/(n+m-2)
    cohen = abs(mx-my)/(math.sqrt(max(pv, 0.0))+1e-32)
    s = 0
    for x in xs:
        for y in ys:
            s += (x > y) - (x < y)
    cliffs = abs(s)/(n*m)
    sx, sy = sorted(xs), sorted(ys)
    d = 0.0
    for v in set(xs) | set(ys):
        d = max(d, abs(bisect_right(sx, v)/n - bisect_right(sy, v)/m))
    ks = d/math.sqrt(1/n + 1/m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": bool(same)}

def main():
    a = sys.argv[1:]
    if not a or a[0] != "certify":
        sys.exit(1)
    path = a[a.index("--test")+1]
    with open(path) as f:
        cases = json.load(f)
    json.dump({"comparisons": [stats([float(x) for x in c["xs"]], [float(y) for y in c["ys"]]) for c in cases]}, sys.stdout)

main()

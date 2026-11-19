import sys, json, math
from bisect import bisect_right

def stats(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs)/n, sum(ys)/m
    vx = sum((x-mx)**2 for x in xs)/(n-1)
    vy = sum((y-my)**2 for y in ys)/(m-1)
    pv = ((n-1)*vx + (m-1)*vy)/(n+m-2)
    cohen = abs(mx-my)/(math.sqrt(pv)+1e-32)
    sx, sy = sorted(xs), sorted(ys)
    # Cliff's via sorted ys
    tot = 0
    for x in xs:
        lt = bisect_right(sy, x - 0) 
        # count y < x and y > x
        lo = 0
        import bisect
        lo = bisect.bisect_left(sy, x)
        hi = bisect.bisect_right(sy, x)
        tot += lo - (m - hi)
    cliffs = abs(tot)/(n*m)
    ks = 0.0
    for v in set(xs) | set(ys):
        d = abs(bisect_right(sx, v)/n - bisect_right(sy, v)/m)
        ks = max(ks, d)
    ks /= math.sqrt(1/n + 1/m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}

def main():
    a = sys.argv[1:]
    if not a or a[0] != "certify":
        sys.exit(1)
    test = a[a.index("--test")+1]
    with open(test) as f:
        cases = json.load(f)
    out = {"comparisons": [stats([float(v) for v in c["xs"]], [float(v) for v in c["ys"]]) for c in cases]}
    sys.stdout.write(json.dumps(out))

main()

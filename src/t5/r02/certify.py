import sys, json
from math import sqrt

def stats(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs)/n, sum(ys)/m
    vx = sum((x-mx)**2 for x in xs)/(n-1)
    vy = sum((y-my)**2 for y in ys)/(m-1)
    pv = ((n-1)*vx + (m-1)*vy)/(n+m-2)
    cohen = abs(mx-my)/(sqrt(pv)+1e-32)
    gt = lt = 0
    for x in xs:
        for y in ys:
            if x > y: gt += 1
            elif x < y: lt += 1
    cliffs = abs(gt-lt)/(n*m)
    sx, sy = sorted(xs), sorted(ys)
    i = j = 0
    d = 0.0
    for v in sorted(set(xs) | set(ys)):
        while i < n and sx[i] <= v: i += 1
        while j < m and sy[j] <= v: j += 1
        d = max(d, abs(i/n - j/m))
    ks = d/sqrt(1/n + 1/m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}

def main():
    a = sys.argv[1:]
    if not a or a[0] != "certify":
        sys.exit(1)
    test = a[a.index("--test")+1]
    with open(test) as f:
        cases = json.load(f)
    out = [stats([float(v) for v in c["xs"]], [float(v) for v in c["ys"]]) for c in cases]
    sys.stdout.write(json.dumps({"comparisons": out}))

main()

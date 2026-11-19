import sys, json, math

def mean(a):
    return sum(a) / len(a)

def svar(a):
    m = mean(a)
    return sum((x - m) ** 2 for x in a) / (len(a) - 1)

def compare(xs, ys):
    n, m = len(xs), len(ys)
    pooled = ((n - 1) * svar(xs) + (m - 1) * svar(ys)) / (n + m - 2)
    cohen = abs(mean(xs) - mean(ys)) / (math.sqrt(pooled) + 1e-32)
    s = 0
    for x in xs:
        for y in ys:
            s += (x > y) - (x < y)
    cliffs = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    i = j = 0
    d = 0.0
    for v in sorted(set(xs) | set(ys)):
        while i < n and sx[i] <= v:
            i += 1
        while j < m and sy[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": bool(same)}

def main():
    a = sys.argv[1:]
    if not a or a[0] != "certify":
        sys.exit(1)
    path = a[a.index("--test") + 1]
    with open(path) as f:
        cases = json.load(f)
    out = {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}
    sys.stdout.write(json.dumps(out))

main()

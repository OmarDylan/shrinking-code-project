import sys, csv, json, math

def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return hdr, rows

def main():
    a = sys.argv
    if len(a) < 2 or a[1] != 'remember':
        sys.exit(1)
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    goals = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]

    def cols(hdr):
        return [hdr.index(g) for g, _ in goals]

    h, tr = read(train)
    ci = cols(h)
    tr = [[float(r[i]) for i in ci] for r in tr if all(r[i] != '?' for i in ci)]
    n = len(tr)
    stats = []
    for k in range(3):
        v = [r[k] for r in tr]
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    h2, te = read(test)
    ci2 = cols(h2)
    scores = []
    for r in te:
        s = 0.0
        for k, i in enumerate(ci2):
            m, sd = stats[k]
            z = max(-3.0, min(3.0, (float(r[i]) - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - goals[k][1]) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(scores) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

main()

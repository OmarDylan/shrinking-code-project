#!/usr/bin/env python3
"""CSC491 T7 candidate: explicitly AI-assisted wrapper around earlier EG5/EG6 homework.

The source modules under prior/ are byte-for-byte copies of the user's uploaded
coursework. This wrapper was produced with AI assistance; it is NOT an unaided
human baseline. Implements all ten skill interfaces; competency must be checked on real data.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remember(train_path, test_path):
    # Reuse EG5's Num.norm(), Tbl.disty(), and csv_rows() unchanged.
    eg5 = load_module('t7_eg5', 'prior/eg5/eg5.py')
    training = eg5.Tbl(eg5.csv_rows(train_path))
    test_rows = iter(eg5.csv_rows(test_path))
    header = next(test_rows)
    if header != training.cols.names:
        raise ValueError('The training and test CSV headers must match.')
    scores = [training.disty(row) for row in test_rows]
    order = sorted(range(len(scores)), key=lambda j: (scores[j], j))
    return {
        'scores': scores,
        'top': order[:10],
        'best': order[0],
        'median': order[len(order) // 2],
        'worst': order[-1],
    }


def certify(_train_path, test_path):
    # Reuse EG6's statistics; unlike EG6.same(), the frozen V2 contract
    # requires a 0.197 Cliff's threshold (EG6's default was 0.195).
    eg6 = load_module('t7_eg6', 'prior/eg6/eg6.py')
    cases = json.loads(Path(test_path).read_text(encoding='utf-8'))
    comparisons = []
    for case in cases:
        xs = sorted(float(v) for v in case['xs'])
        ys = sorted(float(v) for v in case['ys'])
        c = eg6.cohen(xs, ys)
        d = eg6.cliffs(xs, ys)
        k = eg6.ks(xs, ys)
        comparisons.append({
            'cohen': c, 'cliffs': d, 'ks': k,
            'same': c <= 0.35 and d <= 0.197 and k <= 1.36,
        })
    return {'comparisons': comparisons}


def predict_from_training(train_path, test_path, streaming=False):
    """EG5 five-neighbor model; all labels and normalization come from training.

    Both Guess and Flow use the exact same fixed model. Only the four input
    columns are used for matching: Clndrs, Volume, Model, and origin.
    """
    eg5 = load_module('t7_predict_eg5', 'prior/eg5/eg5.py')
    t = eg5.Tbl(eg5.csv_rows(train_path))
    mpg_at = t.cols.names.index('Mpg+')
    training_scores = [t.disty(row) for row in t.rows]
    history = sorted(training_scores)
    test_rows = iter(eg5.csv_rows(test_path))
    if next(test_rows) != t.cols.names:
        raise ValueError('The training and test CSV headers must match.')

    if streaming:
        eg6 = load_module('t7_flow_eg6', 'prior/eg6/eg6.py')
    mpg, scores, alerts = [], [], []
    for row in test_rows:
        # Tbl.distx() works only with input columns and handles missing Volume.
        neighbors = sorted(
            range(len(t.rows)),
            key=lambda j: (round(t.distx(row, t.rows[j]), 12), j)
        )[:5]
        mpg.append(max(1e-12, sum(t.rows[j][mpg_at] for j in neighbors) / len(neighbors)))
        estimate = sum(training_scores[j] for j in neighbors) / len(neighbors)
        scores.append(max(0.0, min(1.0, estimate)))
        if streaming:
            if len(scores) < 20:
                alerts.append(False)
            else:
                recent = sorted(scores[-20:])
                c = eg6.cohen(history, recent)
                d = eg6.cliffs(history, recent)
                k = eg6.ks(history, recent)
                alerts.append(not (c <= 0.35 and d <= 0.197 and k <= 1.36))

    out = {'mpg': mpg, 'scores': scores}
    if streaming:
        out['alerts'] = alerts
    return out


def guess(train_path, test_path):
    return predict_from_training(train_path, test_path)


def flow(train_path, test_path):
    return predict_from_training(train_path, test_path, streaming=True)


# --- Remaining T7 tasks: AI-assisted adapters reusing prior EG5/EG6 coursework ---
from math import exp, sqrt
import statistics as stats
import sys

GOALS = {'Lbs-': 0, 'Acc+': 1, 'Mpg+': 1}
FEATURES = ('Clndrs', 'Volume', 'Model', 'origin')


def _data(train_path, test_path=None):
    eg5 = load_module('t7_tasks_eg5', 'prior/eg5/eg5.py')
    table = eg5.Tbl(eg5.csv_rows(train_path))
    if test_path is None:
        return eg5, table, None
    reader = iter(eg5.csv_rows(test_path))
    if next(reader) != table.cols.names:
        raise ValueError('The training and test CSV headers must match.')
    return eg5, table, list(reader)


def _eg6():
    return load_module('t7_tasks_eg6', 'prior/eg6/eg6.py')


def _same_stats(xs, ys, eg6=None):
    eg6 = _eg6() if eg6 is None else eg6
    xs, ys = sorted(xs), sorted(ys)
    c, d, k = eg6.cohen(xs, ys), eg6.cliffs(xs, ys), eg6.ks(xs, ys)
    return {'cohen': c, 'cliffs': d, 'ks': k,
            'same': c <= 0.35 and d <= 0.197 and k <= 1.36}


def _rank_dict(values):
    return sorted(values, key=lambda k: (-round(values[k], 10), k))


def _diagnose(table, test_rows):
    # Full-training normalization; EG5 Num objects hold sample means and SDs.
    names = table.cols.names
    by_name = {col.name: col for col in table.cols.all}
    known = [table.disty(row) for row in table.rows]
    ordered = sorted(range(len(known)), key=lambda j: (round(known[j], 12), j))
    q = max(1, len(ordered) // 4)
    good = [table.rows[j] for j in ordered[:q]]
    poor = [table.rows[j] for j in ordered[-q:]]

    def center(group, name):
        at = names.index(name)
        values = [float(row[at]) for row in group if row[at] != '?']
        return by_name[name].norm(stats.mean(values))

    def frequency(group, name, value):
        at = names.index(name)
        vals = [row[at] for row in group if row[at] != '?']
        return vals.count(value) / len(vals) if vals else 0.0

    centers = {
        name: (center(good, name), center(poor, name))
        for name in FEATURES if name[0].isupper()
    }
    results = []
    for row in test_rows:
        gaps = {
            name: abs(by_name[name].norm(row[names.index(name)]) - heaven)
            for name, heaven in GOALS.items()
        }
        evidence = {}
        for name in FEATURES:
            value = row[names.index(name)]
            if value == '?':
                raw = 0.0
            elif name in centers:
                value_norm = by_name[name].norm(value)
                gc, pc = centers[name]
                raw = abs(value_norm - gc) - abs(value_norm - pc)
            else:
                raw = frequency(poor, name, value) - frequency(good, name, value)
            evidence[name] = max(0.0, raw)
        results.append({
            'goals': gaps, 'features': evidence,
            'goal_order': _rank_dict(gaps),
            'feature_order': _rank_dict(evidence),
        })
    return results


def blame(train_path, test_path):
    _, table, test = _data(train_path, test_path)
    return {'diagnoses': _diagnose(table, test)}


def justify(train_path, test_path):
    _, table, test = _data(train_path, test_path)
    known = sorted(table.disty(row) for row in table.rows)
    cutoff = known[len(known) // 2]
    scores = [table.disty(row) for row in test]
    diagnoses = _diagnose(table, test)
    results = []
    for j, a in enumerate(scores):
        k = (j + 1) % len(test)
        b = scores[k]
        accepted = round(a, 12) <= round(cutoff, 12)
        winner = None if round(a, 12) == round(b, 12) else (j if a < b else k)
        advantage = None
        if winner is not None:
            loser = k if winner == j else j
            gains = {
                name: diagnoses[loser]['goals'][name] ** 2
                - diagnoses[winner]['goals'][name] ** 2
                for name in GOALS
            }
            advantage = _rank_dict(gains)[0]
        current = diagnoses[j]
        weakest = current['goal_order'][0]
        feature = current['feature_order'][0]
        if current['features'][feature] <= 1e-10:
            feature = None
        verdict = 'accepted' if accepted else 'rejected'
        explanation = (f'Car {j} {verdict}: score {a:.3f}; '
                       f'other score {b:.3f}; cutoff {cutoff:.3f}. '
                       f'Weakest goal: {weakest}.')
        if winner is None:
            explanation += ' The two cars tie.'
        else:
            explanation += f' Recommend car {winner}; advantage from {advantage}.'
        if feature is not None:
            explanation += f' Associated input evidence: {feature}.'
        results.append({
            'row': j, 'other': k, 'score': a, 'other_score': b,
            'cutoff': cutoff, 'accepted': accepted, 'winner': winner,
            'advantage_goal': advantage, 'weakest_goal': weakest,
            'top_feature': feature, 'text': explanation,
        })
    return {'explanations': results}


def _knn_score(table, observed_scores, row):
    neighbors = sorted(
        range(len(table.rows)),
        key=lambda j: (round(table.distx(table.rows[j], row), 12), j)
    )[:5]
    return sum(observed_scores[j] for j in neighbors) / len(neighbors)


def _oracle_request(message):
    print(json.dumps(message, separators=(',', ':')), flush=True)
    response = sys.stdin.readline()
    if not response:
        raise EOFError('Oracle stopped before responding')
    return json.loads(response)


def _observed_score(table, row, goals):
    observed = list(row)
    for name in GOALS:
        observed[table.cols.names.index(name)] = float(goals[name])
    return table.disty(observed)


def choose(train_path, test_path):
    _, table, test = _data(train_path, test_path)
    observed_train = [table.disty(row) for row in table.rows]
    predicted = [_knn_score(table, observed_train, row) for row in test]
    # Query only the 10 most promising available test rows, then sort by oracle labels.
    order = sorted(range(len(test)), key=lambda j: (predicted[j], j))[:10]
    labeled = {}
    for j in order:
        reply = _oracle_request({'inspect': j})
        if reply.get('row') != j:
            raise ValueError('Unexpected oracle row')
        labeled[j] = _observed_score(table, test[j], reply['goals'])
    selected = sorted(order, key=lambda j: (round(labeled[j], 12), j))[:3]
    return {'selected': selected}


def fix(train_path, test_path):
    _, table, test = _data(train_path, test_path)
    known = [table.disty(row) for row in table.rows]
    cutoff = sorted(known)[len(known) // 2]
    ranked = sorted(range(len(known)), key=lambda j: (round(known[j], 12), j))
    edit_cols = ('Clndrs', 'Volume')
    menu, seen = [], set()
    for j in ranked[:max(1, len(known) // 4)]:
        pair = tuple(float(table.rows[j][table.cols.names.index(x)]) for x in edit_cols)
        if pair not in seen:
            seen.add(pair)
            menu.append(pair)
        if len(menu) >= 12:
            break
    output = []
    for j, row in enumerate(test):
        before = _knn_score(table, known, row)
        answer = {'row': j, 'before': before, 'after': before,
                  'changes': {}, 'status': 'already_good'}
        if round(before, 10) > round(cutoff, 10):
            answer['status'] = 'no_candidate'
            candidates = []
            for pair in menu:
                changed = {}
                edited = list(row)
                for name, val in zip(edit_cols, pair):
                    idx = table.cols.names.index(name)
                    if float(row[idx]) != val:
                        changed[name] = val
                        edited[idx] = val
                if not changed:
                    continue
                after = _knn_score(table, known, edited)
                if round(before - after, 10) >= 0.05:
                    key = (len(changed), round(table.distx(row, edited), 10),
                           round(after, 10), pair)
                    candidates.append((key, after, changed))
            if candidates:
                _, after, changed = min(candidates, key=lambda c: c[0])
                answer.update(after=after, changes=changed, status='improved')
        output.append(answer)
    return {'repairs': output}


def spot(train_path, test_path):
    _, table, test = _data(train_path, test_path)
    # An unsupervised training-only anomaly score; no synthetic corruption labels
    # or held-out goals are used in fitting its threshold.
    names = table.cols.names
    indices = [(name, names.index(name)) for name in FEATURES]
    numeric = {
        name: (table.cols.all[idx].mid(), table.cols.all[idx].div())
        for name, idx in indices if name[0].isupper()
    }
    known_cats = {name: {row[idx] for row in table.rows if row[idx] != '?'}
                  for name, idx in indices if name[0].islower()}

    def anomaly(row):
        parts = []
        for name, idx in indices:
            value = row[idx]
            if value == '?':
                parts.append(0.6)
            elif name in numeric:
                mu, sd = numeric[name]
                z = abs((float(value) - mu) / (sd + 1e-32))
                parts.append(min(1.0, z / 6.0))
            else:
                parts.append(0.0 if value in known_cats[name] else 1.0)
        return max(parts)

    historical = sorted(anomaly(row) for row in table.rows)
    # Fit a high-quantile cutoff on training scores; retain headroom for unseen inputs.
    threshold = min(0.99, max(0.10, historical[int(0.98 * (len(historical) - 1))] + 0.025))
    scores = [anomaly(row) for row in test]
    return {'scores': scores, 'alerts': [s > threshold for s in scores],
            'threshold': threshold}


def race(train_path, test_path):
    eg5, table, _ = _data(train_path)
    eg6 = _eg6()
    payload = json.loads(Path(test_path).read_text(encoding='utf-8'))
    if payload['header'] != table.cols.names:
        raise ValueError('Unexpected race header')
    rows = [[eg5.thing(str(v)) for v in row] for row in payload['rows']]
    training_scores = [table.disty(row) for row in table.rows]
    known = sorted(training_scores)
    target = known[len(known) // 4]
    # All initial estimates depend only on the training dataset and masked inputs.
    base = [_knn_score(table, training_scores, row) for row in rows]
    results = []
    for e, episode in enumerate(payload['episodes']):
        record = {'episode': e}
        for strategy in ('adaptive', 'random'):
            pool = episode['pool']
            inspected = []
            observed = {}
            for step in range(10):
                if strategy == 'random':
                    j = episode['random_order'][step]
                elif not inspected:
                    j = min(pool, key=lambda idx: (base[idx], idx))
                else:
                    # Adaptive residual correction uses only inspections from this
                    # episode and this strategy; never other episodes or oracle runs.
                    def rank(idx):
                        nearest = min(inspected, key=lambda old: table.distx(rows[idx], rows[old]))
                        distance = table.distx(rows[idx], rows[nearest])
                        residual = observed[nearest] - base[nearest]
                        corrected = base[idx] + 0.6 * residual * exp(-4 * distance)
                        return (corrected, idx)
                    j = min((idx for idx in pool if idx not in observed), key=rank)
                reply = _oracle_request({'episode': e, 'strategy': strategy,
                                         'inspect': j})
                if reply.get('row') != j:
                    raise ValueError('Unexpected oracle row in race')
                inspected.append(j)
                observed[j] = _observed_score(table, rows[j], reply['goals'])
            values = [observed[idx] for idx in inspected]
            record[strategy] = {
                'best5': min(values[:5]), 'best10': min(values),
                'labels_to_target': next(
                    (i+1 for i, val in enumerate(values)
                     if round(val, 12) <= round(target, 12)), 11),
            }
        results.append(record)
    metrics = ('best5', 'best10', 'labels_to_target')
    comparisons = {
        metric: _same_stats([r['adaptive'][metric] for r in results],
                            [r['random'][metric] for r in results], eg6)
        for metric in metrics
    }
    return {'results': results, 'comparisons': comparisons}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('skill', choices=['remember', 'guess', 'certify', 'flow', 'blame', 'justify', 'choose', 'fix', 'spot', 'race'])
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    args = parser.parse_args()
    result = {'remember': remember, 'guess': guess, 'certify': certify,
              'flow': flow, 'blame': blame, 'justify': justify,
              'choose': choose, 'fix': fix, 'spot': spot,
              'race': race}[args.skill](args.train, args.test)
    print(json.dumps(result, allow_nan=False, separators=(',', ':')))


if __name__ == '__main__':
    main()

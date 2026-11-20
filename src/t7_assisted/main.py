#!/usr/bin/env python3
"""CSC491 T7 candidate: explicitly AI-assisted wrapper around earlier EG5/EG6 homework.

The source modules under prior/ are byte-for-byte copies of the user's uploaded
coursework. This wrapper was produced with AI assistance; it is NOT an unaided
human baseline. Implemented tasks so far: remember, guess, certify, and flow.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('skill', choices=['remember', 'guess', 'certify', 'flow'])
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    args = parser.parse_args()
    result = {'remember': remember, 'guess': guess, 'certify': certify, 'flow': flow}[args.skill](args.train, args.test)
    print(json.dumps(result, allow_nan=False, separators=(',', ':')))


if __name__ == '__main__':
    main()

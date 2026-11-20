#!/usr/bin/env python3
"""Run the ten unchanged CSC491 V2 checkers against the AI-assisted T7 candidate.

Runs generated source in a local Python subprocess. Review source before use.
Writes one immutable timestamped result folder with logs and JSON summaries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SKILLS = 'remember guess certify flow blame justify choose fix spot race'.split()
PROGRAM = ROOT / 'src/t7_assisted/main.py'
PRIOR = [ROOT / 'src/t7_assisted/prior/eg5/eg5.py',
         ROOT / 'src/t7_assisted/prior/eg6/eg6.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default=None,
                   help='New folder for logs; must not already exist')
    args = p.parse_args()
    paths = [PROGRAM, *PRIOR, *[ROOT / 'tests' / f'check_{s}.py' for s in SKILLS]]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        p.error('Missing required source/checkers: ' + ', '.join(missing))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path(args.output_dir) if args.output_dir else ROOT / 'experiments/t7-assisted' / f'local-{stamp}'
    folder = folder.resolve()
    if folder.exists():
        p.error('Refusing to overwrite existing evidence: ' + str(folder))
    folder.mkdir(parents=True)

    env = os.environ.copy()
    env.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    provenance = {
        'treatment': 'T7-assisted',
        'protocol': 'v2',
        'condition': 'coursework reuse, AI-assisted new adapters',
        'not_human_only': True,
        'repetition': 1,
        'dataset': str(ROOT / 'data/splits/seed1'),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': {str(path.relative_to(ROOT)): {
            'sha256': digest(path), 'physical_lines': len(path.read_text().splitlines())}
            for path in [PROGRAM, *PRIOR]},
        'human_development_time_seconds': None,
        'ai_assistance_time_seconds': None,
        'generation_tokens': None,
        'api_equivalent_cost_usd': None,
    }
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    rows = []
    for skill in SKILLS:
        print(f'\n=== T7 assisted {skill} ===', flush=True)
        payload = folder / f'{skill}.json'
        cmd = [sys.executable, str(ROOT / 'tests' / f'check_{skill}.py'),
               str(PROGRAM), str(payload)]
        start = time.perf_counter()
        try:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                    text=True, timeout=130, env=env)
            code = result.returncode
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            code = 124
            stdout = exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
            stderr = 'Checker exceeded 130-second timeout\n'
        wall = time.perf_counter() - start
        (folder / f'{skill}.txt').write_text(stdout + '\nSTDERR:\n' + stderr)
        status = 'PASS' if code == 0 and re.search(r'(?m)^PASS(?:\s|$)', stdout) else 'FAIL'
        metric = None
        if payload.exists():
            try:
                metric = json.loads(payload.read_text())
            except (ValueError, OSError):
                pass
        execution = None
        if isinstance(metric, dict):
            execution = metric.get('runtime', (metric.get('summary') or {}).get('runtime'))
        if execution is None:
            match = re.search(r'Runtime:\s*([0-9.]+)s', stdout)
            if match: execution = float(match.group(1))
        rows.append({'skill': skill, 'status': status, 'returncode': code,
                     'checker_wall_seconds': round(wall, 6),
                     'reported_execution_seconds': execution,
                     'result_file': payload.name if payload.exists() else None})
        print(stdout.strip()[-2000:], flush=True)
        if status != 'PASS':
            print('CHECK FAILED; details:', folder / f'{skill}.txt', flush=True)
            if stderr: print(stderr[-1200:], file=sys.stderr, flush=True)
            break
    summary = {'provenance': provenance, 'checks': rows, 'passed': sum(r['status']=='PASS' for r in rows),
               'required': len(SKILLS), 'complete': len(rows)==len(SKILLS) and all(r['status']=='PASS' for r in rows)}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('\nEvidence:', folder)
    print('Skills passed:', summary['passed'], '/', summary['required'])
    if not summary['complete']:
        raise SystemExit('At least one check failed; evidence retained. Do not report T7 as fully passing.')
    print('All ten V2 competency checkers PASSED.')
    print('This is an AI-assisted T7 observation, not an unaided human-only baseline.')


if __name__ == '__main__':
    main()

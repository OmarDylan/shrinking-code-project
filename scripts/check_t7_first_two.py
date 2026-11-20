#!/usr/bin/env python3
"""Execute frozen Remember and Certify competency checkers for AI-assisted T7 candidate."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / 'src' / 't7_assisted' / 'main.py'


def main():
    if not PROGRAM.is_file():
        raise SystemExit(f'Candidate not found: {PROGRAM}')
    for check in ('remember', 'certify'):
        print(f'\n=== T7 assisted {check} ===', flush=True)
        cmd = [sys.executable, str(ROOT / 'tests' / f'check_{check}.py'), str(PROGRAM)]
        env = os.environ.copy()
        env.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        run = subprocess.run(cmd, cwd=ROOT, env=env, timeout=120)
        if run.returncode:
            raise SystemExit(f'{check}: FAIL (return code {run.returncode})')
    print('\nTwo competencies checked; eight others remain unimplemented.', flush=True)


if __name__ == '__main__':
    main()

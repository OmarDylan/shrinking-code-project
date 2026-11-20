#!/usr/bin/env python3
"""Check four implemented T7 AI-assisted skills against unchanged frozen V2 tests."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / 'src' / 't7_assisted' / 'main.py'


def main():
    if not PROGRAM.is_file():
        raise SystemExit('T7 assisted program missing.')
    env = os.environ.copy()
    env.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for skill in ('remember', 'guess', 'certify', 'flow'):
        print(f'\n=== T7 assisted {skill} ===', flush=True)
        run = subprocess.run(
            [sys.executable, str(ROOT / 'tests' / f'check_{skill}.py'), str(PROGRAM)],
            cwd=ROOT, env=env, timeout=180,
        )
        if run.returncode:
            raise SystemExit(f'CHECK FAILED: {skill} (exit {run.returncode})')
    print('\nAll four implemented T7 skills passed; six skills remain.', flush=True)


if __name__ == '__main__':
    main()

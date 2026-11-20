#!/usr/bin/env python3
"""Evaluate four existing V2 repetitions via a single restricted GitHub Action.

Never creates or modifies a generated snapshot and never invokes Claude/DeepSeek.
Requires explicit review before executing untrusted AI-generated source in Docker.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_v2 import RECORDED, classify_record, folder

ROOT = Path(__file__).resolve().parents[1]
REPO = 'OmarDylan/shrinking-code-project'
WORKFLOW = 'csc491-v2-evaluate-recorded.yml'
PATCH = [
    '.github/workflows/csc491-v2-evaluate-recorded.yml',
    'scripts/evaluate_recorded_v2.py',
    'docs/recorded-v2-evaluation.md',
]


def command(*args, capture=False):
    a = [str(x) for x in args]
    if capture:
        return subprocess.check_output(a, cwd=ROOT, text=True).strip()
    subprocess.run(a, cwd=ROOT, check=True)


def git(*args):
    return command('git', *args, capture=True)


def validate_snapshots():
    text = [
        'CSC491 V2 frozen source-review manifest: r02 to r05',
        'Untrusted generated programs. Review code before APPROVING cloud execution.',
        '',
    ]
    for rep in range(2, 6):
        states = [classify_record(t, s, rep) for t in range(1, 7) for s in range(1, 11)]
        if len(states) != 60 or set(states) - RECORDED:
            raise SystemExit(f'R{rep:02d} incomplete or unresolved. Refusing to evaluate.')
        for t in range(1, 7):
            for s in range(1, 11):
                path = folder(t, s, rep) / 'generated.py'
                state = classify_record(t, s, rep)
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else '-'
                text.append(f'R{rep:02d} T{t} S{s:02d} {state:15s} {path.relative_to(ROOT)} sha256={digest}')
    path = ROOT / '.git/csc491-r02-r05-review.txt'
    path.write_text('\n'.join(text) + '\n')
    return path


def lookup(sha):
    raw = command('gh', 'run', 'list', '--repo', REPO, '--workflow', WORKFLOW,
                  '--limit', '50', '--json', 'databaseId,displayTitle,headSha,status,conclusion', capture=True)
    found = [x for x in json.loads(raw) if x['headSha'] == sha and
             x['displayTitle'] == f'CSC491 recorded R02-R05 {sha}']
    return max(found, key=lambda x: x['databaseId']) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviewed', action='store_true', help='You have reviewed all generated snapshots')
    args = ap.parse_args()
    for binary in ('git', 'gh', 'docker'):
        if not shutil.which(binary):
            raise SystemExit(f'{binary} not found. Install it or check PATH.')
    if git('branch', '--show-current') != 'main':
        raise SystemExit('Stay on the current main branch; no new branch needed.')
    if git('remote', 'get-url', 'origin').rstrip('/').removesuffix('.git') != f'https://github.com/{REPO}':
        raise SystemExit('Unexpected origin URL. No push performed.')
    if command('gh', 'repo', 'view', REPO, '--json', 'visibility', '--jq', '.visibility', capture=True) != 'PRIVATE':
        raise SystemExit('Expected a private repository. No push performed.')
    if git('diff', '--cached', '--name-only'):
        raise SystemExit('Staged changes exist. Please commit or unstage them first.')
    for p in PATCH:
        if not (ROOT / p).is_file():
            raise SystemExit(f'Missing patch file: {p}')
    manifest = validate_snapshots()
    command('git', 'add', '--', *PATCH)
    if git('diff', '--cached', '--name-only'):
        command('git', 'commit', '-m', 'Evaluate recorded V2 repetitions without new AI calls')
    sha = git('rev-parse', 'HEAD')
    print(f'Generated source-review manifest: {manifest}')
    if not args.reviewed:
        print('Review the source files listed above. Re-run with --reviewed once done.')
        return
    confirm = input('Type REVIEWED to approve restricted cloud execution: ').strip()
    if confirm != 'REVIEWED':
        raise SystemExit('Review not approved; no source executed.')
    command('git', 'push', 'origin', 'main')
    remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    if remote != sha:
        raise SystemExit('Remote snapshot mismatch; evaluation not launched.')
    record = lookup(sha)
    if record is None:
        command('gh', 'workflow', 'run', WORKFLOW, '--repo', REPO, '--ref', 'main', '-f', f'snapshot={sha}')
        for _ in range(30):
            time.sleep(3)
            record = lookup(sha)
            if record:
                break
    if record is None:
        raise SystemExit('Workflow not visible. Inspect GitHub Actions before running again.')
    rid = str(record['databaseId'])
    print(f'GitHub Actions: https://github.com/{REPO}/actions/runs/{rid}', flush=True)
    if record['status'] == 'completed' and record['conclusion'] != 'success':
        raise SystemExit(f'Action run {rid} failed; see gh run view {rid} --log-failed')
    command('gh', 'run', 'watch', rid, '--repo', REPO, '--exit-status')
    dest = ROOT / 'experiments/downloads' / f'recorded-r02-r05-{rid}'
    dest.mkdir(parents=True, exist_ok=True)
    artifact = dest / 'csc491-v2-recorded-r02-r05'
    if not artifact.is_dir():
        command('gh', 'run', 'download', rid, '--repo', REPO, '--dir', dest)
    # gh download may flatten the sole artifact into dest.
    report = next(iter(sorted(dest.rglob('REPORT.md'))), None)
    if report is None:
        raise SystemExit(f'No REPORT.md found in {dest}; inspect downloaded artifacts.')
    print('\n' + report.read_text()[:25000])
    archive = shutil.make_archive(str(Path.home() / 'Desktop/CSC491_R02_R05_EVALUATED'), 'zip', dest)
    print(f'\nArchive ready for analysis: {archive}', flush=True)


if __name__ == '__main__':
    main()

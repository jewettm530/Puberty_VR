#!/usr/bin/env python3
"""Run the decided RT200 screen, then recompute learning only in eligible pairs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--known-noncompliant',type=int,nargs='*',default=[34])
    p.add_argument('--permutations',type=int,default=19999)
    p.add_argument('--bootstraps',type=int,default=3000)
    p.add_argument('--skip-hr',action='store_true')
    a=p.parse_args();root=a.project_root.resolve();scripts=Path(__file__).resolve().parent
    if not (root/'data/processed/plearning').is_dir():
        p.error('Missing data/processed/plearning. Place this package in the project or pass --project-root /path/to/Puberty_VR')
    logs=root/'outputs/learning_rates/pipeline_logs';logs.mkdir(parents=True,exist_ok=True)
    common=['--project-root',str(root)]
    steps=[('engagement_screen',[str(scripts/'screen_task_engagement.py'),*common,'--min-rt-ms','200',
            '--known-noncompliant',*[str(i) for i in a.known_noncompliant]]),
        ('learning_analysis',[str(scripts/'analyze_rt200_learning.py'),*common,'--permutations',str(a.permutations),
            '--bootstraps',str(a.bootstraps),*(['--skip-hr'] if a.skip_hr else [])])]
    env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','1')
    for name,argv in steps:
        log=logs/f'{name}.log'
        print(f'Running {name}; progress log: {log}',flush=True)
        with log.open('w') as handle:
            result=subprocess.run([sys.executable,*argv],stdout=handle,stderr=subprocess.STDOUT,env=env)
        if result.returncode:
            print('\n'.join(log.read_text().splitlines()[-25:]),file=sys.stderr)
            raise SystemExit(result.returncode)
        print(f'Completed {name}.',flush=True)
    print('Reports: outputs/learning_rates/task_engagement_screen_rt200/{exclusion_criteria,excluded_subjects_report,engagement_screen_report}.txt',flush=True)
    print('Learning summary: outputs/learning_rates/rt200_learning_analysis/learning_analysis_summary.txt',flush=True)


if __name__=='__main__':main()

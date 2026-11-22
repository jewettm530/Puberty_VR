#!/usr/bin/env python3
"""Run task-engagement/RL checks using only the pipeline dependencies."""
import runpy
from pathlib import Path

base=Path(__file__).resolve().parent
count=0
for filename in ['test_task_engagement_screen.py','test_rt200_learning_analysis.py','test_learning_pipeline_reports.py']:
    namespace=runpy.run_path(str(base/filename))
    for name,value in namespace.items():
        if name.startswith('test_') and callable(value):
            value();count+=1;print('PASS',name)
print(f'{count} checks passed.')

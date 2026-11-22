#!/usr/bin/env python3
"""Apply the user's retrospective task-engagement rules, without accuracy filtering.

Per session: >20% missed choices; >25% scheduled trials removed specifically
for RT below 200 ms; dominant response key >90% (also report
>95%). Subject 034 is excluded independently based on the user's report;
the specific affected session is unknown. Thresholds are strict >, not >=.
This is an operational eligibility screen, not a validated motivation diagnosis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd


def screen_session(d, min_rt_ms=200.):
    d = d.copy().sort_values(['blockNumber', 'trialNumber']).reset_index(drop=True)
    choice = d.chosen.astype('string').str.strip().str.lower()
    key = d.response_choice_p.astype('string').str.strip().str.lower()
    valid = choice.isin(['face', 'house'])
    key_valid = key.isin(['left', 'right'])
    rt = pd.to_numeric(d.response_time, errors='coerce')
    usable_rt = valid & rt.notna() & np.isfinite(rt) & (rt >= 0)
    if (valid & ~key_valid).any() or (~valid & key_valid).any():
        raise ValueError('Choice and response-key validity do not match')
    if (valid & ~usable_rt).any():
        raise ValueError('A completed choice has an invalid RT; inspect before screening')
    median = rt[usable_rt].median()
    sd = rt[usable_rt].std(ddof=1)
    # Median and SD are descriptive only; no upper RT exclusion is applied.
    low, high = min_rt_ms, np.inf
    outlier = usable_rt & (rt < min_rt_ms)
    counts = key[key_valid].value_counts()
    dominant = counts.iloc[0] / key_valid.sum() if len(counts) else np.nan
    # Secondary diagnostic: consecutive same-key responding within blocks.
    previous = key.shift(1)
    adjacent = key_valid & key_valid.shift(1, fill_value=False) & d.blockNumber.eq(d.blockNumber.shift(1))
    repetitions = ((key == previous).fillna(False) & adjacent).sum()
    n = len(d)
    missed = ~valid
    retained = valid & ~outlier
    row = dict(n_scheduled=n, n_completed=int(valid.sum()), n_missed=int(missed.sum()),
        missed_fraction=float(missed.mean()), median_rt_ms=median, sample_sd_rt_ms=sd,
        rt_lower_ms=low, rt_upper_ms=high, n_rt_outliers=int(outlier.sum()),
        rt_outlier_fraction_scheduled=float(outlier.mean()),
        rt_outlier_fraction_completed=float(outlier.sum()/valid.sum()),
        n_total_unusable=int((missed | outlier).sum()),
        total_unusable_fraction=float((missed | outlier).mean()),
        n_retained=int(retained.sum()), dominant_key=counts.index[0] if len(counts) else '',
        n_dominant_key=int(counts.iloc[0]) if len(counts) else 0,
        dominant_key_fraction=dominant,
        consecutive_key_repeat_fraction=float(repetitions/adjacent.sum()) if adjacent.any() else np.nan,
        flag_missed_gt20=bool(missed.mean() > .20),
        flag_rt_removed_gt25=bool(outlier.mean() > .25),
        flag_key_gt90=bool(dominant > .90), flag_key_gt95=bool(dominant > .95),
        flag_total_unusable_gt25_sensitivity=bool((missed | outlier).mean() > .25),
        n_zero_rt=int((usable_rt & rt.eq(0)).sum()),
        fraction_rt_under150_completed=float((rt[usable_rt] < 150).mean()))
    row['behavioral_pass'] = not any(row[x] for x in ['flag_missed_gt20','flag_rt_removed_gt25','flag_key_gt90'])
    d['missed_trial'] = missed
    d['rt_below_lower_bound'] = outlier
    d['retained_trial'] = retained
    return row, d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--known-noncompliant', type=int, nargs='*', default=[34])
    p.add_argument('--min-rt-ms', type=float, default=200.)
    args = p.parse_args()
    root = args.project_root
    if not np.isfinite(args.min_rt_ms) or args.min_rt_ms <= 0:
        raise ValueError('Minimum RT must be a positive finite number')
    out = root/f'outputs/learning_rates/task_engagement_screen_rt{args.min_rt_ms:g}'
    out.mkdir(parents=True, exist_ok=True)
    sessions, trials, fingerprints = [], [], {}
    for path in sorted((root/'data/processed/plearning').glob('*_plearning_[12].csv')):
        subject, session = map(int, re.fullmatch(r'(\d+)_plearning_([12])\.csv', path.name).groups())
        d = pd.read_csv(path)
        fingerprints[f'{subject:03d}:{session}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(d) != 180 or d.blockNumber.nunique() != 10 or d.duplicated(['blockNumber','trialNumber']).any():
            raise ValueError(f'Unexpected trial structure: {path}')
        for _, block in d.groupby('blockNumber'):
            if len(block) != 18 or set(block.trialNumber) != set(range(1,19)):
                raise ValueError(f'Unexpected block structure: {path}')
        row, t = screen_session(d, args.min_rt_ms)
        row.update(subject_id=f'{subject:03d}', session=session,
                   documented_noncompliance=subject in args.known_noncompliant)
        row['session_behaviorally_usable'] = row['behavioral_pass']
        row['analysis_eligible'] = row['behavioral_pass'] and not row['documented_noncompliance']
        row['reason'] = '; '.join(label for flag,label in [
            ('documented_noncompliance','reported noncompliance; session unspecified'),
            ('flag_missed_gt20','>20% missed'), ('flag_rt_removed_gt25',f'>25% responses below {args.min_rt_ms:g} ms'),
            ('flag_key_gt90','>90% same response key')] if row[flag]) or 'passes requested screen'
        t.insert(0, 'session', session); t.insert(0, 'subject_id', f'{subject:03d}')
        t['session_analysis_eligible'] = row['analysis_eligible']
        sessions.append(row); trials.append(t)
    s = pd.DataFrame(sessions)
    if s.empty:
        raise ValueError('No session CSV files found under data/processed/plearning')
    persons = []
    for subject,a in s.groupby('subject_id', sort=True):
        both = set(a.session) == {1,2}
        passed = bool(a.analysis_eligible.all())
        persons.append(dict(subject_id=subject, sessions_present=','.join(map(str,sorted(a.session))),
            has_both_sessions=both, all_available_sessions_pass=passed,
            eligible_for_paired_analysis=both and passed,
            failed_sessions=','.join(map(str,a.loc[~a.analysis_eligible,'session'])),
            reasons='; '.join(sorted(set(a.loc[~a.analysis_eligible,'reason']))) or 'passes requested screen'))
    people = pd.DataFrame(persons)
    all_trials = pd.concat(trials, ignore_index=True)
    s.to_csv(out/'session_engagement_screen.csv',index=False)
    people.to_csv(out/'participant_eligibility.csv',index=False)
    all_trials.to_csv(out/'all_trials_with_screen_flags.csv',index=False)
    all_trials.loc[all_trials.retained_trial & all_trials.session_analysis_eligible].to_csv(
        out/'retained_trials_eligible_sessions.csv',index=False)
    paired_ids = set(people.loc[people.eligible_for_paired_analysis,'subject_id'])
    all_trials.loc[all_trials.retained_trial & all_trials.subject_id.isin(paired_ids)].to_csv(
        out/'retained_trials_paired_cohort.csv',index=False)
    paired = people[people.has_both_sessions]
    lines = [__doc__,
        'RT median and sample SD (ddof=1) are descriptive only, calculated from completed choices.',
        f'RT exclusion: completed choices with RT <{args.min_rt_ms:g} ms. Exactly {args.min_rt_ms:g} ms is retained; no upper cutoff.',
        'RT removal denominator: all 180 scheduled trials; omissions have their separate >20% rule.',
        'Same-key fraction: dominant left/right response / all completed choices, BEFORE RT trimming.',
        'Both key thresholds reported; >90% is the operative rule because it subsumes >95%.',
        'Secondary consecutive-repeat diagnostic is not the primary same-key rule.',
        'Documented subject-level exclusions: '+(', '.join(f'{v:03d}' for v in args.known_noncompliant) or 'none')+'; session scope unspecified.',
        'No accuracy, learning slope, learning rate, HR, or first/last-half change used for exclusion.',
        f'Sessions: {len(s)}; eligible {int(s.analysis_eligible.sum())}; excluded {int((~s.analysis_eligible).sum())}.',
        f'All subjects: {len(people)}; pass all available {int(people.all_available_sessions_pass.sum())}; fail {int((~people.all_available_sessions_pass).sum())}.',
        f'Paired subjects: {len(paired)}; eligible {int(paired.eligible_for_paired_analysis.sum())}; excluded {int((~paired.eligible_for_paired_analysis).sum())}.',
        'Paired eligible IDs: '+', '.join(paired.loc[paired.eligible_for_paired_analysis,'subject_id']),
        'Paired excluded IDs: '+', '.join(paired.loc[~paired.eligible_for_paired_analysis,'subject_id']),
        '\nFlagged sessions:\n'+s.loc[~s.analysis_eligible,['subject_id','session','reason','missed_fraction',
            'rt_outlier_fraction_scheduled','dominant_key_fraction']].to_string(index=False),
        '\nCriterion counts (may overlap):\n'+s[['flag_missed_gt20','flag_rt_removed_gt25','flag_key_gt90','flag_key_gt95',
            'flag_total_unusable_gt25_sensitivity']].sum().to_string(),
        '\nThese criteria define retrospective analysis eligibility; passing does not prove motivation.',
        'The lower-bound rule replaces median +/-2 SD. It is an adaptation of published procedures, not a validated CPLearn motivation test.',
        'Retained-trial exports are supplied. Existing model fits have NOT been rerun.',
        'Future RL fits must preserve outcomes/sequence history across removed observations; do not stitch gaps as adjacent trials.']
    report='\n\n'.join(lines)+'\n'
    (out/'engagement_screen_report.txt').write_text(report)
    write_exclusion_reports(out, s, people, args.min_rt_ms, args.known_noncompliant, fingerprints)
    print(report)


def write_exclusion_reports(out, sessions, people, min_rt_ms, known, fingerprints):
    """Write explicit rules and participant/session reasons as readable text."""
    criteria=[
        'TASK-ENGAGEMENT EXCLUSION CRITERIA',
        'Apply every rule separately within each experimental session (10 blocks x 18 trials = 180 scheduled trials).',
        '1. Documented noncompliance: exclude listed subjects independently of behavioral flags. Current IDs: '+(', '.join(f'{i:03d}' for i in known) or 'none')+'. Subject 034 is based on the user-reported admission; no affected session was specified.',
        '2. Missed responses: exclude session if missed choices / scheduled trials >0.20. With 180 trials: 37 or more omissions. Completed choices are face/house with a matching valid left/right response key.',
        f'3. Rapid responses: remove completed choices with RT <{min_rt_ms:g} ms. Exactly {min_rt_ms:g} ms is retained. Exclude session if rapid responses / all scheduled trials >0.25. With 180 trials: 46 or more rapid responses. Missed trials are counted separately, not added to this rapid-response exclusion numerator. No upper RT or median/SD-based RT cutoff is used.',
        '4. Extreme key bias: exclude session if the most-used left/right key / all completed choices >0.90, before RT removal. This is dominant-key frequency, not consecutive-key repetition or face/house repetition. Exactly 90% passes. The >95% flag is reported, but is redundant under >90%.',
        'Paired analyses require sessions 1 and 2, both passing, and no documented subject-level noncompliance. A participant with one failed session is excluded from all paired comparisons. Passing single sessions may contribute only to session-specific outlier reference distributions.',
        'Missing session 2 is an incomplete pair, not proof of non-engagement. Neither low accuracy nor a significant speed/accuracy change across block halves is an engagement exclusion.',
        'Computational outliers are a separate step: a one-pass absolute z-score >=2 removes values only from relevant comparisons, not all analyses for that person. Fit-boundary exclusions are additional sensitivity checks.',
        'Literature-informed adaptations: Pereg et al. (2024), doi:10.1038/s41539-024-00214-0 (absolute RT screening, >25% removed, >90% same key); Donegan et al. (2023), doi:10.1038/s44271-023-00031-y (>20% missed, >95% same key); Senta et al. (2025), doi:10.1371/journal.pcbi.1012872 (attention/compliance and timeout checks); Bradshaw & McGettigan (2021), doi:10.1177/1747021821999663 (omissions and instruction adherence). These studies did not all use the same cutoffs. Bradshaw & McGettigan abandoned a 200-ms cutoff in their different task design.',
        'These retrospective rules define analysis eligibility; they are not a validated diagnosis of motivation or random responding.'
    ]
    (out/'exclusion_criteria.txt').write_text('\n\n'.join(criteria)+'\n')
    paired=people[people.has_both_sessions]
    lines=['SUBJECT EXCLUSION AUDIT',
        f'All available participants: {len(people)}; pass every available session: {int(people.all_available_sessions_pass.sum())}; fail: {int((~people.all_available_sessions_pass).sum())}.',
        f'Complete pairs: {len(paired)}; included: {int(paired.eligible_for_paired_analysis.sum())}; excluded: {int((~paired.eligible_for_paired_analysis).sum())}.',
        'A failed session excludes a subject from paired analysis even when their other session passes.']
    for r in people.loc[~people.all_available_sessions_pass].itertuples():
        lines.append(f'\nSUBJECT {r.subject_id}: EXCLUDED FROM PAIRED ANALYSIS' if r.has_both_sessions else f'\nSUBJECT {r.subject_id}: FAILED SCREEN; ALSO MISSING A PAIRED SESSION')
        for a in sessions[sessions.subject_id==r.subject_id].itertuples():
            lines.append(f'  Session {a.session}: {"PASS" if a.analysis_eligible else "FAIL"}. {a.reason}. '
                f'Missed {a.n_missed}/{a.n_scheduled} ({100*a.missed_fraction:.2f}%); '
                f'RT <{min_rt_ms:g} ms {a.n_rt_outliers}/{a.n_scheduled} ({100*a.rt_outlier_fraction_scheduled:.2f}%); '
                f'dominant key {a.dominant_key}: {a.n_dominant_key}/{a.n_completed} ({100*a.dominant_key_fraction:.2f}%).')
    incomplete=people[~people.has_both_sessions]
    lines.extend(['\nPARTICIPANTS WITHOUT BOTH SESSIONS (SEPARATE FROM SCREEN FAILURE):',
        ', '.join(incomplete.subject_id) or 'none',
        '\nINCLUDED PAIRED PARTICIPANTS:',', '.join(paired.loc[paired.eligible_for_paired_analysis,'subject_id'])])
    (out/'excluded_subjects_report.txt').write_text('\n'.join(lines)+'\n')
    (out/'included_subjects.txt').write_text('PAIRED ANALYSIS IDS\n'+
        '\n'.join(paired.loc[paired.eligible_for_paired_analysis,'subject_id'])+'\n')
    config=dict(min_rt_ms=min_rt_ms,missed_fraction_exclude_above=.20,rapid_fraction_exclude_above=.25,
        rapid_fraction_denominator='all scheduled trials',dominant_key_exclude_above=.90,
        dominant_key_denominator='completed choices before RT removal',documented_noncompliance_ids=known,
        strict_greater_than=True,input_sha256=fingerprints)
    (out/'screen_configuration.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__ == '__main__':
    main()

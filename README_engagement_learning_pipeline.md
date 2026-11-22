# RT200 engagement screening and learning analysis

This package contains the current scripts, their required helpers, tests, and reports. It reproduces the decided rules without the earlier first-half/last-half deterioration screen or median +/-2 SD RT rule. Place/merge its `scripts`, `tests`, and documentation files into your existing Puberty_VR project. Original processed or raw data are not modified. The included output files are a snapshot of the verified run; rerunning replaces these pipeline outputs.

## Run everything

Use Python 3.10 or newer. From your project root:

```bash
python3 -m pip install -r requirements_learning_pipeline.txt
python3 scripts/learning_rates/run_engagement_learning_pipeline.py
```

The runner first regenerates screening, then recomputes learning from the included cohort. It writes progress logs under `outputs/learning_rates/pipeline_logs/`. Statistical resampling and model refitting can take several minutes. Nothing is hard-coded to require 31 participants; the current data produce 31 complete included pairs.

To run learning without supplemental HR:

```bash
python3 scripts/learning_rates/run_engagement_learning_pipeline.py --skip-hr
```

HR is supplemental and automatically skipped if the processed HR file is unavailable. It is never used for participant eligibility. To reproduce the previously reported supplemental correlations, use the corrected processed HR summary described below and do not use `--skip-hr`.

To run from another location:

```bash
python3 /path/to/Puberty_VR/scripts/learning_rates/run_engagement_learning_pipeline.py --project-root /path/to/Puberty_VR
```

Documented noncompliance defaults to subject 034, based on your reported admission. To set a different documented list, supply `--known-noncompliant 34 99` with actual documented IDs. Supply `--known-noncompliant` with no IDs to clear that list. Do not add IDs merely because they learn poorly. The runner rebuilds cohort membership from the session screen each time.

## Scripts included

- `scripts/learning_rates/run_engagement_learning_pipeline.py`: one-command entry point; runs steps in order and stops on errors.
- `scripts/learning_rates/screen_task_engagement.py`: session and subject eligibility, trial flags, exact criteria, inclusion/exclusion reports, and input fingerprints.
- `scripts/learning_rates/analyze_rt200_learning.py`: recomputes behavioral indices, fits asymmetric RL and exponential curves, performs paired comparisons and all correlation families, and generates figures.
- `scripts/learning_rates/learning_analysis_helpers.py`: shared Spearman permutation/bootstrap and Holm/BH procedures.
- `scripts/learning_rates/learning_analysis_reports.py`: generates readable TXT interpretation and summary directly from the current result tables.

These files do not depend on the earlier 26-person analysis scripts or their output tables. Older project analyses can remain in place, but they do not use this eligibility cohort automatically. Use this runner for the updated analyses.

## Required inputs

- `data/processed/plearning/NNN_plearning_1.csv`
- `data/processed/plearning/NNN_plearning_2.csv`

Each file must contain ten experimental blocks of 18 original trials, with columns `winner`, `blockNumber`, `trialNumber`, `chosen`, `feedback`, `response_time` (milliseconds), and `response_choice_p` (`left`/`right` or no response). Do not pre-delete trials from these input CSVs: the screening and fitting scripts need the original trial sequence and denominators. Practice rows must already have been removed by the project's preprocessing step.

Optional supplemental HR input: `data/analysis_data/heart_rate/heart_rate_summary_wide.csv`, containing `subject_id`, `pre_vr_mean_hr`, `vr_prep_mean_hr`, `vr_speech_mean_hr`, `vr_math_mean_hr`, and the prep/speech/math reactivity columns. Use the corrected HR output; this package does not extract HR from raw sensor files.

## Exact screening decisions

Apply each rule within each session:

1. Exclude documented noncompliance at subject level (default 034; affected session unspecified).
2. Exclude sessions missing **>20%** of scheduled trials: at least 37 of 180.
3. Remove completed responses with **RT <200 ms**. Exactly 200 ms is retained. Exclude sessions with **>25%** scheduled trials removed for this reason: at least 46 of 180. The numerator includes fast responses only; omissions are governed by rule 2.
4. Exclude sessions where one response key accounts for **>90% of completed choices**, measured before RT removal. This is dominant left/right key frequency, not face/house preference or the percentage of consecutive repeated presses. A >95% flag is also recorded but does not add exclusions under the >90% rule.

For paired learning analysis, both sessions must pass and the participant must have no documented noncompliance. Missing sessions are reported separately from non-engagement. For these data: 38 complete pairs, 31 included and 7 excluded (005, 018, 029, 032, 034, 039, 054). Across all 46 participants, eight fail their available-session screen, including single-session 033.

These are retrospective, literature-informed operational rules, not a validated determination of participants' motivation. Thresholds are adapted from multiple studies rather than copied as one common validated procedure. Exclusion reports include the study references and explain which elements informed each criterion.

## Reports to read

Under `outputs/learning_rates/task_engagement_screen_rt200/`:

- `exclusion_criteria.txt`: exact rules, strict boundary comparisons, denominators, paired eligibility, and source studies.
- `excluded_subjects_report.txt`: every excluded participant, their session-level counts/percentages, overlapping reasons, incomplete pairs, and included IDs.
- `included_subjects.txt`: eligible paired subject IDs.
- `engagement_screen_report.txt`: screening overview and diagnostic tables.
- `screen_configuration.json`: machine-readable criteria and SHA256 hashes of input CSVs. The learning script refuses to use a stale screen if inputs changed.

Under `outputs/learning_rates/rt200_learning_analysis/`:

- **`learning_analysis_summary.txt`**: human-readable results and interpretations; mean/SD/change/CI, raw and corrected p-values, significant correlations, sensitivity results, boundaries, outlier IDs, acquisition and half-session tests, and optional HR.
- `learning_analysis_report.txt`: expanded numerical audit and manifest.
- `all_correlations.csv`: every tested association, including nonsignificant ones, sample sizes, IDs, bootstrap CIs, raw p and adjusted q.
- `paired_changes.csv`: all pre/post comparisons, SDs, change CIs, individual directions, and corrections.
- `outlier_audit.csv`, `paired_cohort_outliers.csv`: reference means/SDs, z-scores and flags.
- `recomputed_session_measures.csv`: refitted parameters and boundary flags for eligible sessions. Single-session references are marked `paired_cohort=False` and never enter paired/correlation analyses.
- Other CSVs: retained trials, RPE replay, subject-level curves, within-session tests, nonboundary sensitivity, and domain-specific correlation tables.

## How learning is calculated

The primary analysis excludes missed choices and responses <200 ms. Linear slope and exponential k use original trial positions, not renumbered retained trials. Win-stay/lose-switch require both current and previous trials retained and originally adjacent within a block; no fabricated transitions are created across removed trials.

The asymmetric RL model estimates alpha+, alpha-, and inverse temperature beta, resetting Q_face and Q_house to 0.5 each block. Fast responses do not contribute to choice likelihood, but their observed choice/outcome still updates Q, since the feedback was experienced. Missing choices do not update Q. This is a modeling assumption, and an all-completed-trial sensitivity analysis is supplied for the same participant cohort.

Computational outliers are flagged once at absolute z >=2, separately by measure, without iterative trimming. Level references use all screen-eligible sessions; change references use the included paired participants. A flag removes only affected comparisons; it does not exclude the person from every analysis. A separate boundary-exclusion sensitivity analysis addresses parameter reliability.

Spearman tests use 19,999 permutations and 3,000 participant bootstrap samples by default, with stable seeds per comparison. BH correction spans 351 core learning pairs among pre/post/change measures. The 279 added RPE pairs and, if run, 189 HR pairs have separate correction families. A broader 630-learning/RPE correction is also exported. Raw p<.05 alone is not the significance criterion for an exploratory correlation. Level-versus-change correlations are explicitly labeled because their components overlap.

Paired tests use Wilcoxon for alpha+, alpha-, beta and k, and paired t tests for other measures. Holm corrections across all nine core outcomes and within smaller families are both given. The lose-switch result must specify which correction family is used. Mean-change and correlation CIs are not multiplicity-adjusted.

## Figures

All PNGs are generated by `analyze_rt200_learning.py`:

- `paired_learning_changes.png`
- `learning_correlation_heatmaps.png`
- `group_average_curves.png`
- `strategy_learning_and_supplemental_hr_scatterplots.png`

If HR is skipped, its scatterplot panel is marked unavailable. All counts and titles are derived from the current included cohort.

## Run the checks

```bash
python3 tests/run_rt200_pipeline_checks.py
```

These checks cover strict threshold boundaries, timeout handling, fixed RT screening, RL gradients and feedback history, strategy adjacency, resampling, and output cohort integrity. No pytest installation is required.

# Run 2026-10-31T01-51-37Z-judge-calibration-vs-labels-claude-code-claude-sonnet-4-6-team-plan

- name: Judge calibration vs labels, Claude Code (claude-sonnet-4-6, Team plan)
- started: 2026-10-31T01:51:37+00:00
- git: 4d876de
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 5%, gpu 28.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 01:51:37Z judging run: 2026-10-31T01-51-32Z-judge-claude-code-claude-sonnet-4-6-team-plan-100-labelled-answers
- 01:51:37Z items judged: 100 of 100
- 01:51:37Z label counts: correct 83, partly 1, wrong 16
- 01:51:37Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[16, 0, 0], [0, 1, 0], [0, 3, 80]]
- 01:51:37Z weighted kappa (linear) 0.947, 95% bootstrap CI 0.870 to 1.000; length vs judge -0.289, length vs labels -0.193
- 01:51:38Z wrote summary.json (5 rows, status Verified)

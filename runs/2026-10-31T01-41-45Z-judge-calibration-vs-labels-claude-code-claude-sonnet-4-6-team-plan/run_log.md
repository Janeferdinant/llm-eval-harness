# Run 2026-10-31T01-41-45Z-judge-calibration-vs-labels-claude-code-claude-sonnet-4-6-team-plan

- name: Judge calibration vs labels, Claude Code (claude-sonnet-4-6, Team plan)
- started: 2026-10-31T01:41:45+00:00
- git: 8100565 (dirty)
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 43%, gpu 29.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 01:41:45Z judging run: 2026-10-31T01-37-23Z-judge-claude-code-claude-sonnet-4-6-team-plan-100-labelled-answers
- 01:41:45Z items judged: 100 of 100
- 01:41:45Z label counts: correct 83, partly 1, wrong 16
- 01:41:45Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[16, 0, 0], [0, 1, 0], [0, 3, 80]]
- 01:41:45Z weighted kappa (linear) 0.947, 95% bootstrap CI 0.870 to 1.000; length vs judge -0.289, length vs labels -0.193
- 01:41:45Z wrote summary.json (5 rows, status Verified)

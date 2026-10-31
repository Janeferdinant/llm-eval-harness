# Run 2026-10-31T01-51-49Z-judge-calibration-vs-labels-codex-cli-gpt-6-1-sol

- name: Judge calibration vs labels, Codex CLI (gpt-6.1-sol)
- started: 2026-10-31T01:51:49+00:00
- git: 4d876de
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 6%, gpu 28.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 01:51:49Z judging run: 2026-10-31T01-51-43Z-judge-codex-cli-gpt-6-1-sol-100-labelled-answers
- 01:51:49Z items judged: 100 of 100
- 01:51:49Z label counts: correct 83, partly 1, wrong 16
- 01:51:49Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[16, 0, 0], [0, 1, 0], [0, 1, 82]]
- 01:51:49Z weighted kappa (linear) 0.982, 95% bootstrap CI 0.935 to 1.000; length vs judge -0.224, length vs labels -0.193
- 01:51:49Z wrote summary.json (5 rows, status Verified)

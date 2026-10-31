# Run 2026-10-31T01-45-27Z-judge-calibration-vs-labels-codex-cli-gpt-6-1-sol

- name: Judge calibration vs labels, Codex CLI (gpt-6.1-sol)
- started: 2026-10-31T01:45:27+00:00
- git: 8100565 (dirty)
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 14%, gpu 9.0

## Log

- 01:45:27Z judging run: 2026-10-31T01-41-51Z-judge-codex-cli-gpt-6-1-sol-100-labelled-answers
- 01:45:27Z items judged: 100 of 100
- 01:45:27Z label counts: correct 83, partly 1, wrong 16
- 01:45:27Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[16, 0, 0], [0, 1, 0], [0, 1, 82]]
- 01:45:27Z weighted kappa (linear) 0.982, 95% bootstrap CI 0.935 to 1.000; length vs judge -0.224, length vs labels -0.193
- 01:45:27Z wrote summary.json (5 rows, status Verified)

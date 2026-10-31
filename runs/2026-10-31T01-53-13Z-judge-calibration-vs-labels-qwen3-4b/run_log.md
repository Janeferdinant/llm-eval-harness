# Run 2026-10-31T01-53-13Z-judge-calibration-vs-labels-qwen3-4b

- name: Judge calibration vs labels, qwen3:4b
- started: 2026-10-31T01:53:13+00:00
- git: 4d876de
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 6%, gpu 27.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 01:53:13Z judging run: 2026-10-31T01-53-06Z-judge-qwen3-4b-100-labelled-answers
- 01:53:13Z items judged: 100 of 100
- 01:53:13Z label counts: correct 83, partly 1, wrong 16
- 01:53:13Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [2, 0, 81]]
- 01:53:13Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.660 to 0.962; length vs judge -0.305, length vs labels -0.193
- 01:53:13Z wrote summary.json (5 rows, status Verified)

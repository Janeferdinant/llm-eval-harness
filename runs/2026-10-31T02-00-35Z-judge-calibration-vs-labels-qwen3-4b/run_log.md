# Run 2026-10-31T02-00-35Z-judge-calibration-vs-labels-qwen3-4b

- name: Judge calibration vs labels, qwen3:4b
- started: 2026-10-31T02:00:35+00:00
- git: beeac1c
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 5%, gpu 9.0

## Log

- 02:00:35Z judging run: 2026-10-31T01-53-06Z-judge-qwen3-4b-100-labelled-answers
- 02:00:35Z items judged: 100 of 100
- 02:00:35Z label counts: correct 83, partly 1, wrong 16
- 02:00:35Z confusion matrix (rows: human label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [2, 0, 81]]
- 02:00:35Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.660 to 0.962; length vs judge -0.305, length vs labels -0.193
- 02:00:35Z wrote summary.json (5 rows, status Verified)

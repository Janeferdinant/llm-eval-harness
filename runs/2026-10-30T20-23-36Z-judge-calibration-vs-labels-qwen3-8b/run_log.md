# Run 2026-10-30T20-23-36Z-judge-calibration-vs-labels-qwen3-8b

- name: Judge calibration vs labels, qwen3:8b
- started: 2026-10-30T20:23:36+00:00
- git: 36432b3 (dirty)
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 9%, gpu 29.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 20:23:36Z judging run: 2026-10-30T20-22-48Z-judge-qwen3-8b-100-labelled-answers
- 20:23:36Z items judged: 100 of 100
- 20:23:36Z label counts: correct 83, partly 1, wrong 16
- 20:23:36Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [0, 4, 79]]
- 20:23:36Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.682 to 0.945; length vs judge -0.279, length vs labels -0.193
- 20:23:37Z wrote summary.json (5 rows, status Verified)

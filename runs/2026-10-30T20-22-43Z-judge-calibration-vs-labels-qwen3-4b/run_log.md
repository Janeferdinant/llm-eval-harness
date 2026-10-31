# Run 2026-10-30T20-22-43Z-judge-calibration-vs-labels-qwen3-4b

- name: Judge calibration vs labels, qwen3:4b
- started: 2026-10-30T20:22:43+00:00
- git: 36432b3 (dirty)
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 4%, gpu 21.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 20:22:43Z judging run: 2026-10-30T20-21-48Z-judge-qwen3-4b-100-labelled-answers
- 20:22:43Z items judged: 100 of 100
- 20:22:43Z label counts: correct 83, partly 1, wrong 16
- 20:22:43Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [2, 0, 81]]
- 20:22:43Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.660 to 0.962; length vs judge -0.305, length vs labels -0.193
- 20:22:43Z wrote summary.json (5 rows, status Verified)

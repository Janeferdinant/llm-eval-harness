# Run 2026-10-31T01-53-24Z-judge-calibration-vs-labels-qwen3-8b

- name: Judge calibration vs labels, qwen3:8b
- started: 2026-10-31T01:53:24+00:00
- git: 4d876de
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 4%, gpu 27.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 01:53:24Z judging run: 2026-10-31T01-53-18Z-judge-qwen3-8b-100-labelled-answers
- 01:53:24Z items judged: 100 of 100
- 01:53:24Z label counts: correct 83, partly 1, wrong 16
- 01:53:24Z confusion matrix (rows: your label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [0, 4, 79]]
- 01:53:24Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.682 to 0.945; length vs judge -0.279, length vs labels -0.193
- 01:53:24Z wrote summary.json (5 rows, status Verified)

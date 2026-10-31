# Run 2026-10-31T02-00-40Z-judge-calibration-vs-labels-qwen3-8b

- name: Judge calibration vs labels, qwen3:8b
- started: 2026-10-31T02:00:40+00:00
- git: beeac1c
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 7%, gpu 8.0

## Log

- 02:00:40Z judging run: 2026-10-31T01-53-18Z-judge-qwen3-8b-100-labelled-answers
- 02:00:40Z items judged: 100 of 100
- 02:00:40Z label counts: correct 83, partly 1, wrong 16
- 02:00:40Z confusion matrix (rows: human label wrong/partly/correct; columns: judge score 0/1/2): [[14, 0, 2], [0, 0, 1], [0, 4, 79]]
- 02:00:40Z weighted kappa (linear) 0.835, 95% bootstrap CI 0.682 to 0.945; length vs judge -0.279, length vs labels -0.193
- 02:00:40Z wrote summary.json (5 rows, status Verified)

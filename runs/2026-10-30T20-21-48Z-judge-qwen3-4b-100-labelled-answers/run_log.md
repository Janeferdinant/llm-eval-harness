# Run 2026-10-30T20-21-48Z-judge-qwen3-4b-100-labelled-answers

- name: Judge: qwen3:4b, 100 labelled answers
- started: 2026-10-30T20:21:48+00:00
- git: 36432b3
- seed: 0
- python: 3.11.14 on Windows-10-10.0.26200-SP0
- load before start: cpu 8%, gpu 36.0
- **busy machine**: timings are not clean (started with allow_busy)

## Log

- 20:21:48Z suite squad2-calibration-labelled v1: 100 cases, 1 repeat(s), concurrency 1
- 20:21:48Z metrics: rubric judge
- 20:21:48Z config: judge = qwen3:4b
- 20:21:48Z config: model = qwen3:4b
- 20:21:48Z config: scale = 0 to 2
- 20:21:48Z config: temperature = 0.0
- 20:21:48Z config: seed = 0
- 20:21:48Z config: max_tokens = 1024
- 20:21:48Z config: model digest = 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
- 20:21:48Z config: num_ctx = 8192
- 20:21:48Z config: think = False
- 20:22:38Z 100 rows written, 0 with errors
- 20:22:38Z wrote summary.json (2 rows, status Verified)

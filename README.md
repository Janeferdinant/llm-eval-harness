# llm-eval-harness

A small Python library and command line tool (`evalh`) for evaluating LLM systems: suites of cases, text and schema metrics, an LLM-as-judge metric that is calibrated against human labels, bootstrap confidence intervals, a CI gate that fails a build when a metric drops, and the model clients the other repos in this series reuse.

## Summary
<!-- summary:start -->
**Token F1 as a judge, against human labels (busy machine)**: Verified (2026-10-31)

Run `2026-10-31T01-55-09Z-token-f1-baseline-vs-labels`, commit `0319ea0`, AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, NVIDIA GeForce RTX 3080.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| token F1 baseline: Spearman with human labels | 0.870 | n/a | 100 |
| token F1 baseline: weighted kappa vs human labels | 0.883 | n/a | 100 |

**Judge calibration vs human labels: qwen3:4b (busy machine)**: Verified (2026-10-31)

Run `2026-10-31T02-00-35Z-judge-calibration-vs-labels-qwen3-4b`, commit `beeac1c`, AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, NVIDIA GeForce RTX 3080.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| weighted kappa vs human labels, qwen3:4b | 0.835 | 0.660 to 0.962 | 100 |
| answer length vs judge score (Spearman), qwen3:4b | -0.305 | n/a | 100 |
| answer length vs human labels (Spearman), qwen3:4b | -0.193 | n/a | 100 |
| latency per judgement, qwen3:4b | 0.491 s | n/a | 100 |
| cost per 100 judgements, API-equivalent (USD), qwen3:4b | 0.000 USD | n/a | 100 |

**Judge calibration vs human labels: qwen3:8b (busy machine)**: Verified (2026-10-31)

Run `2026-10-31T02-00-40Z-judge-calibration-vs-labels-qwen3-8b`, commit `beeac1c`, AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, NVIDIA GeForce RTX 3080.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| weighted kappa vs human labels, qwen3:8b | 0.835 | 0.682 to 0.945 | 100 |
| answer length vs judge score (Spearman), qwen3:8b | -0.279 | n/a | 100 |
| answer length vs human labels (Spearman), qwen3:8b | -0.193 | n/a | 100 |
| latency per judgement, qwen3:8b | 0.423 s | n/a | 100 |
| cost per 100 judgements, API-equivalent (USD), qwen3:8b | 0.000 USD | n/a | 100 |

**Judge calibration vs human labels: Claude Code (claude-sonnet-4-6, Team plan) (busy machine)**: Verified (2026-10-31)

Run `2026-10-31T02-00-46Z-judge-calibration-vs-labels-claude-code-claude-sonnet-4-6-team-plan`, commit `beeac1c`, AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, NVIDIA GeForce RTX 3080.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| weighted kappa vs human labels, Claude Code (claude-sonnet-4-6, Team plan) | 0.947 | 0.870 to 1.000 | 100 |
| answer length vs judge score (Spearman), Claude Code (claude-sonnet-4-6, Team plan) | -0.289 | n/a | 100 |
| answer length vs human labels (Spearman), Claude Code (claude-sonnet-4-6, Team plan) | -0.193 | n/a | 100 |
| latency per judgement, Claude Code (claude-sonnet-4-6, Team plan) | 4.846 s | n/a | 100 |
| cost per 100 judgements, API-equivalent (USD), Claude Code (claude-sonnet-4-6, Team plan) | 3.216 USD | n/a | 100 |

**Judge calibration vs human labels: Codex CLI (gpt-6.1-sol) (busy machine)**: Verified (2026-10-31)

Run `2026-10-31T02-00-51Z-judge-calibration-vs-labels-codex-cli-gpt-6-1-sol`, commit `beeac1c`, AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, NVIDIA GeForce RTX 3080.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| weighted kappa vs human labels, Codex CLI (gpt-6.1-sol) | 0.982 | 0.935 to 1.000 | 100 |
| answer length vs judge score (Spearman), Codex CLI (gpt-6.1-sol) | -0.224 | n/a | 100 |
| answer length vs human labels (Spearman), Codex CLI (gpt-6.1-sol) | -0.193 | n/a | 100 |
| latency per judgement, Codex CLI (gpt-6.1-sol) | 4.144 s | n/a | 100 |
| cost per 100 judgements, API-equivalent (USD), Codex CLI (gpt-6.1-sol) | 3.419 USD | n/a | 100 |
<!-- summary:end -->

## What this shows
Status: **v0.1.0.** The judge calibration study has been run (Summary above, Verified 2026-10-30 and 2026-10-31). CI (GitHub Actions: ruff, `pytest`, and the README summary check, all on mocks) passed on 2026-10-31.

What the Summary shows: how closely an LLM judge agrees with human labels on 100 question answering answers, for two local models, two subscription command line tools, and a token F1 baseline that needs no model. Each row is Verified by the run named under it. Nothing else in this README states a measured number.

What exists and is tested (Verified by `uv run pytest` and by CI on mocks, 2026-10-31):
- Suites and cases (`evalh run`), exact match and token F1 with SQuAD normalisation, JSON-schema validity, and a rubric judge.
- Bootstrap confidence intervals, paired deltas, weighted Cohen's kappa, Spearman correlation.
- A regression gate (`evalh gate`) that fails when a metric drops, goes missing or is NaN, or when the error rate rises.
- One client interface with these clients: Ollama, OpenAI-compatible servers (vLLM, llama.cpp), the Claude Code and Codex command line tools, and the OpenAI, Anthropic and Gemini SDKs.
- A run log, a results file and a summary for every run, with the README Summary generated from the runs (CI fails if it drifts).

## Quick start
```bash
uv sync
uv run pytest
uv run evalh run suites/demo --system examples.demo_system:answer --metrics exact_match,token_f1 --out runs/ --allow-busy
uv run evalh gate --baseline tests/fixtures/demo_baseline --candidate runs/<the run folder> --max-drop 0.02
```
`examples/demo_system.py` replays recorded answers, so the demo needs no model and no network.

## Reproduce
- Local models are served by Ollama (`qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, Q4_K_M). Every request sets `num_ctx` explicitly (default 8192): Ollama's own default of 256K for some Qwen3 tags does not fit a 10 GB GPU. Exact model digests go in each run's `run_log.md`.
- `uv run python calibration/prepare.py fetch | answers | sheet | import-labels` prepares the judge calibration set (see `calibration/LABELLING.md`).
- Hosted arms run only through the Claude Code and Codex command line tools, from an empty folder with no user settings, hooks, plugins or MCP servers, and with API keys removed from the environment. Claude Code runs with its built-in tools off, and its start-up event is checked: a tool, MCP server, skill or plugin that the run did not allow stops the run. Codex has no switch for its built-in tools, so its event stream is checked after the run and any tool use fails it (read-only sandbox). Runs also stop if Claude Code reports an API-key source or overage billing, and billing and routing environment variables are removed. Claude Code's auto-memory is not switched off by a flag; it is empty in the empty working folder, and a probe for a word that appears only in the global CLAUDE.md came back negative.

## Design note
**Why calibrate a judge.** An LLM judge is a measuring instrument, and its agreement with human judgement has to be measured on the task before its scores gate a build. The study uses 100 questions from the SQuAD 2.0 development set (60 answerable, 40 unanswerable, sampled with seed 0). `qwen3:8b` answered each one from its gold paragraph, and the repo author labelled each answer correct, partly correct or wrong (the counts are in each calibration run's `run_log.md`). Every judge then graded the same stored answers on a 0 to 2 scale.

**Why weighted kappa.** It measures agreement above chance on an ordinal scale, where being one step off counts for less than two. Here it carries a caveat: only one answer was labelled "partly correct", so the weighting changes almost nothing and the comparison is close to two-class. The 95% interval is a bootstrap over the 100 items.

**What the table supports.** Every judge was compared with the same labels. The intervals of the four LLM judges overlap pairwise, so 100 items cannot rank them. The hosted judges' point estimates are higher than the local ones. The token F1 baseline, which uses no model, scores in the same range as the local judges, so for short factual answers a cheap metric already agrees well with the human labels. Most answers were correct, which makes agreement easier than it would be on a harder task, and the result says nothing about longer free-form answers.

**Verbosity check.** The answers are short, so their length varies little. The correlations between length and score are small for the judges and for the labels, and this check has little power on this set.

**How the numbers were produced.** Local judges ran with temperature 0 and seed 0 (see Limitations on reproducibility). The two command line tools accept neither, so hosted runs are not controlled for them. Every run is labelled "busy machine": desktop programs kept the GPU above the idle limit during the judging, so the latency figures are not clean timings. The calibration summaries were rebuilt from the judging runs' recorded results through the response caches, with no extra model calls; each run's log names the judging run it came from.

**Commit hashes.** The commit hashes in the Summary and in the run logs refer to the development history, which was squashed into a single commit before the repo was published. The run records are kept as they were produced.

**Subscription arms.** Hosted results are "via Claude Code on a Team subscription" and "via Codex CLI on a ChatGPT Plus subscription". They measure the model plus each tool's own system prompt, and the latency includes the tool's start-up. Cost is the API-equivalent price of the tokens used (Claude Code's own figure, and tokens times the published price for Codex), not money paid.

## Data and licences
- Original work: CC BY-NC-ND 4.0 (`LICENSE`).
- Calibration questions come from the SQuAD 2.0 development set (CC BY-SA 4.0). The repo commits only question IDs and the labels, never the SQuAD text.
- Local models: Qwen3 1.7B, 4B and 8B (Apache-2.0), run through Ollama.
- See `NOTICE.md` for components and terms of use.

## Limitations
- The OpenAI, Anthropic and Gemini SDK clients are tested against mocks only. They were never run against the live APIs, and the mock responses are built from the installed SDKs' own response types, not recorded from live calls.
- Hosted models are measured through subscription command line tools, not through the raw APIs. Their numbers include the tool's own system prompt (about 6,000 input tokens for Claude Code and about 17,000 for Codex on a one-word prompt) and start-up time, and their cost is the API-equivalent price, not money paid. Codex cost counts cached input at the full price, so it is an upper bound.
- Neither command line tool accepts a temperature, a seed or a token limit, so hosted runs are not controlled for those. The Anthropic SDK (2.3.0) has no temperature or seed parameter either.
- Claude Code results are labelled "via Claude Code on a Team subscription". Codex does not report which model it used, so the judge arm pins `gpt-6.1-sol`.
- There is no Gemini subscription arm: Gemini CLI no longer serves Google AI Pro accounts, and the Antigravity CLI terms rule out scripted use (see `NOTICE.md`). The Gemini SDK client exists and is mock-tested.
- The calibration set is 100 items, so its confidence intervals are wide.
- Greedy local decoding (temperature 0, seed 0) is not bit-reproducible here: a third run of the same 100 questions on the same machine differed from the first in 4 answers (3 only in letter case, 1 in content), although a second run was identical. The labelled answers are therefore stored (in the git-ignored labelling sheet) and every judge grades those exact answers instead of regenerating them.

## Status labels
- **Verified (YYYY-MM-DD)**: reproduced by a run in `runs/` on that date. The Summary above is generated from those runs.
- **Reported**: taken from a source, not reproduced here.

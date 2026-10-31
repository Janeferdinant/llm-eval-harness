# Notice

Original work in this repo is licensed CC BY-NC-ND 4.0 (see `LICENSE`).

## Third-party components

| Component | Source | Licence | Checked |
|---|---|---|---|
| SQuAD 2.0 (development set; only question IDs and labels are committed) | https://rajpurkar.github.io/SQuAD-explorer/ | CC BY-SA 4.0 | 2026-10-30 |
| Qwen3 1.7B, 4B, 8B (run locally through Ollama) | https://huggingface.co/Qwen | Apache-2.0 | 2026-10-30 |
| openai, anthropic, google-genai Python SDKs (pinned) | PyPI | Apache-2.0 (package metadata) | 2026-10-30 |
| Ollama | https://ollama.com | MIT | 2026-10-30 |

Calibration runs: `results.jsonl` of runs on SQuAD questions can hold short spans copied from SQuAD paragraphs by the model, so those files are git-ignored. Only each run's `run_log.md` and `summary.json` are committed.

## Terms of use for the command line arms

Checked 2026-10-30. These are my readings of the published text, not legal advice. Each arm runs the provider's own, unmodified command line tool on one machine, with the tool's own login, for one person's experiments on public data.

| Arm | Finding | Decision |
|---|---|---|
| Claude Code (`claude -p`) | The Claude Code docs say use is governed by the Commercial Terms for Team, Enterprise and API users, and the Consumer Terms for Free, Pro, Max and Ultra (https://code.claude.com/docs/en/legal-and-compliance). Running the unmodified binary with its own sign-in is the documented path; the conditions there target products that embed or resell it. Neither the Consumer Terms (https://www.anthropic.com/legal/consumer-terms) nor the Commercial Terms (https://www.anthropic.com/legal/commercial-terms) contain a clause I found against scripted use or against publishing results. The Consumer Terms bar using the Services to train competing models, and this repo never does. | **Built.** The account used is on a **Team plan**, so the Commercial Terms apply; results are labelled "via Claude Code on a Team subscription". |
| Codex CLI (`codex exec`) | OpenAI's own non-interactive-mode docs document `codex exec --json` and sign-in with a ChatGPT account for automation (https://developers.openai.com/codex/noninteractive). The Terms of Use page could not be fetched (HTTP 403 on 2026-10-30), and secondary sources quote a clause against "automatically or programmatically extract data or Output" in the consumer terms; the developer terms are described as narrower. I found no clause on publishing comparisons. | **Built**, with the open point recorded: the consumer-terms wording could not be read first-hand. |
| Gemini CLI / Antigravity CLI (`agy`) | Gemini CLI stopped serving Google AI Pro and individual accounts on 2026-06-18 (`IneligibleTierError` on 2026-10-30). The Antigravity terms (https://antigravity.google/terms) say: "Using third party software, tools, or services to access the Service ... is a breach of this Agreement. Such actions may be grounds for suspension or termination of your Antigravity and/or Gemini CLI accounts." A harness that scripts `agy` is third-party software using the Service. | **Dropped.** `get_client("gemini_cli")` raises `ArmUnavailableError` with this reason. The Gemini SDK client (pay-per-token) is still built and mock-tested. |

Account details (names, addresses, organisation) are never written to run logs, fixtures or this repo; `scripts/prepublish_check.sh` checks for them.

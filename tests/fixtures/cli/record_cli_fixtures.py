"""Record real Claude Code and Codex CLI outputs for the tests (a few tiny calls on the logged-in subscriptions).

    uv run python tests/fixtures/cli/record_cli_fixtures.py

Each call runs from an empty temp folder with the clean-run flags. Paths and the user name are replaced and
thinking signatures are dropped before anything is written. Check the files before committing them.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
ECHO = str(HERE / "echo_mcp.py")
PY = sys.executable
from evalharness.llm.cli import strict_schema  # noqa: E402
from evalharness.metrics.judge import JudgeVerdict  # noqa: E402

SCHEMA = strict_schema(JudgeVerdict.model_json_schema())  # the exact schema the judge sends
PROMPTS = {
    "text": "Reply with the single word OK",
    "structured": 'Return JSON with score set to 3 and reasoning set to "ok".',
    "agent": "Call the echo tool with text 'ping', then reply with exactly what it returned.",
}


def sanitize(text: str) -> str:
    user = os.environ.get("USERNAME", "user")
    for form in (r"C:\\Users\\" + user, r"C:/Users/" + user, r"/c/Users/" + user, "C:\\Users\\" + user):
        text = text.replace(form, "<HOME>")
    text = re.sub(r'"signature":\s*"[^"]*"', '"signature": ""', text)
    text = text.replace(user, "user")
    return text


def run(argv: list[str], prompt: str, env_extra: dict | None = None) -> str:
    exe = shutil.which(argv[0])
    with tempfile.TemporaryDirectory() as empty:
        r = subprocess.run([exe, *argv[1:]], input=prompt, capture_output=True, text=True, encoding="utf-8",
                           cwd=empty, env={**os.environ, **(env_extra or {})}, timeout=300)
    return r.stdout


def save(name: str, cli: str, argv: list[str], stdout: str, **extra) -> None:
    version = subprocess.run([shutil.which(cli), "--version"], capture_output=True, text=True).stdout.strip()
    body = {"recorded_at": date.today().isoformat(), "cli": cli, "cli_version": version, "synthetic": False,
            "argv": [sanitize(a) for a in argv], "stdout": sanitize(stdout), **extra}
    (HERE / f"{name}.json").write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", name, len(stdout), "bytes")


def claude() -> None:
    base = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--tools", "",
            "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence"]
    save("claude_text", "claude", base, run(base, PROMPTS["text"]))
    structured = [*base, "--json-schema", json.dumps(SCHEMA)]
    save("claude_structured", "claude", structured, run(structured, PROMPTS["structured"]), schema=SCHEMA)
    cfg = json.dumps({"mcpServers": {"tools": {"command": PY, "args": [ECHO]}}})
    agent = [*base, "--mcp-config", cfg, "--allowedTools", "mcp__tools__echo", "--max-turns", "4"]
    save("claude_agent", "claude", agent, run(agent, PROMPTS["agent"], {"ENABLE_TOOL_SEARCH": "false"}))


def codex() -> None:
    base = ["codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
            "--ignore-rules", "-s", "read-only", "-"]
    save("codex_text", "codex", base, run(base, PROMPTS["text"]))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(SCHEMA, f)
    structured = [*base[:-1], "--output-schema", f.name, "-"]
    save("codex_structured", "codex", structured, run(structured, PROMPTS["structured"]), schema=SCHEMA)
    mcp = ["-c", f"mcp_servers.tools.command={json.dumps(PY)}", "-c", f"mcp_servers.tools.args={json.dumps([ECHO])}",
           "-c", 'mcp_servers.tools.default_tools_approval_mode="approve"',
           "-c", 'mcp_servers.tools.enabled_tools=["echo"]']
    agent = [*base[:-1], *mcp, "-"]
    save("codex_agent", "codex", agent, run(agent, PROMPTS["agent"]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["claude", "codex"]
    if "claude" in which:
        claude()
    if "codex" in which:
        codex()

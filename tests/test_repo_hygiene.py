import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "out_of_sync_repo"


def _bash() -> str:
    """Git for Windows' bash; plain `bash` on Windows can be the WSL launcher."""
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if sys.platform == "win32" and git_bash.exists():
        return str(git_bash)
    return shutil.which("bash") or "bash"


def _sync(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "sync_readme.py"), *args],
                          cwd=cwd, capture_output=True, text=True)


def test_readme_in_sync():
    assert subprocess.run([sys.executable, "scripts/sync_readme.py", "--check"]).returncode == 0


def test_readme_out_of_sync_fails():
    r = subprocess.run([sys.executable, "scripts/sync_readme.py", "--check",
                        "--root", "tests/fixtures/out_of_sync_repo"])
    assert r.returncode == 1


def test_sync_renders_label_rows_and_ci(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root)
    assert _sync("--root", str(root)).returncode == 0
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Verified (2026-01-01)" in readme
    assert "token F1" in readme and "0.810" in readme and "0.790" in readme and "800" in readme
    assert _sync("--check", "--root", str(root)).returncode == 0


def test_sync_reported_run_has_no_date_in_label(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root)
    summary = root / "runs" / "2026-01-01T00-00-00Z-demo" / "summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["status"] = "Reported"
    summary.write_text(json.dumps(data), encoding="utf-8")
    _sync("--root", str(root))
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Reported" in readme and "Verified" not in readme


def test_sync_fails_when_published_run_is_missing(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root)
    (root / "runs" / "published.txt").write_text("no-such-run\n", encoding="utf-8")
    r = _sync("--check", "--root", str(root))
    assert r.returncode == 1 and "no-such-run" in (r.stdout + r.stderr)


def test_sync_fails_without_markers(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root)
    (root / "README.md").write_text("# no markers here\n", encoding="utf-8")
    assert _sync("--check", "--root", str(root)).returncode == 1


def test_gitignore_keeps_docs_data():
    assert subprocess.run(["git", "check-ignore", "-q", "docs/data/example.json"]).returncode == 1  # 1 = not ignored


def test_gitignore_still_ignores_top_level_data_and_run_artifacts():
    for path in ("data/x.csv", "runs/abc/artifacts/big.bin", ".env"):
        assert subprocess.run(["git", "check-ignore", "-q", path]).returncode == 0, path


def _make_repo(tmp_path: Path, *, notice: bool, extra: dict[str, str] | None = None) -> Path:
    root = tmp_path / "pub"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "prepublish_check.sh", root / "scripts" / "prepublish_check.sh")
    (root / "LICENSE").write_text("licence text\n", encoding="utf-8")
    if notice:
        (root / "NOTICE.md").write_text("notice\n", encoding="utf-8")
    (root / ".prepublish-ignore").write_text("scripts/prepublish_check.sh\n", encoding="utf-8")
    for name, text in (extra or {}).items():
        (root / name).write_text(text, encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    import os
    full = {**os.environ, **env}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, env=full)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=full)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True, env=full)
    return root


def _prepublish(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([_bash(), "scripts/prepublish_check.sh"], cwd=root, capture_output=True, text=True)


def test_prepublish_fails_when_notice_missing(tmp_path):
    r = _prepublish(_make_repo(tmp_path, notice=False))
    assert r.returncode == 1 and "NOTICE.md" in r.stdout + r.stderr


def test_prepublish_flags_a_personal_email_address(tmp_path):
    root = _make_repo(tmp_path, notice=True, extra={"notes.txt": "contact person@acme-corp.net\n"})
    r = _prepublish(root)
    assert r.returncode == 1 and "notes.txt" in r.stdout + r.stderr


def test_prepublish_flags_key_like_token(tmp_path):
    bad = _make_repo(tmp_path, notice=True, extra={"k.txt": "key = sk-abcdefghijklmnop\n"})  # prepublish-allow
    r = _prepublish(bad)
    assert r.returncode == 1 and "k.txt" in r.stdout + r.stderr


def test_prepublish_ignores_ordinary_words_that_contain_sk_dash(tmp_path):
    ok = _make_repo(tmp_path, notice=True, extra={"words.txt": "a risk-free task-list and desk-top\n"})
    r = _prepublish(ok)
    assert "words.txt" not in r.stdout + r.stderr


def test_prepublish_flags_file_over_50_mb(tmp_path):
    big = _make_repo(tmp_path, notice=True)
    with open(big / "big.bin", "wb") as f:
        f.truncate(51 * 1024 * 1024)
    r = _prepublish(big)
    assert r.returncode == 1 and "big.bin" in r.stdout + r.stderr


def test_prepublish_marker_and_ignore_file_allow_a_line(tmp_path):
    root = _make_repo(tmp_path, notice=True, extra={"n.txt": "ok person@acme-corp.net  # prepublish-allow\n"})
    assert "n.txt" not in (_prepublish(root).stdout)


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")
def test_prepublish_passes_on_clean_repo(tmp_path):
    r = _prepublish(_make_repo(tmp_path, notice=True))
    assert r.returncode == 0, r.stdout + r.stderr


def test_prepublish_allows_noreply_and_example_addresses(tmp_path):
    text = "a@users.noreply.github.com noreply@anthropic.com me@example.com 70085962+x@users.noreply.github.com\n"
    root = _make_repo(tmp_path, notice=True, extra={"authors.txt": text})
    assert "authors.txt" not in _prepublish(root).stdout


def test_prepublish_reads_private_patterns_from_an_ignored_file(tmp_path):
    root = _make_repo(tmp_path, notice=True, extra={"about.md": "built at Some Private Corp last year\n"})
    assert "about.md" not in _prepublish(root).stdout  # nothing private configured: not flagged
    (root / ".prepublish-private").write_text("# names that must never be published\nsome private corp\n",
                                              encoding="utf-8")
    r = _prepublish(root)
    assert r.returncode == 1 and "about.md" in r.stdout and "private pattern" in r.stdout
    assert "some private corp" not in r.stdout.lower()  # the finding names the file and line, not the pattern


def test_the_script_and_its_tests_contain_no_personal_strings():
    needles = ["lgeo" + "rge", "arm" + "strong", "li" + "jo"]  # built from parts so this file does not contain them
    for path in (REPO / "scripts" / "prepublish_check.sh", REPO / "tests" / "test_repo_hygiene.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not [n for n in needles if n in text], path.name


def test_prepublish_also_reads_a_shared_private_file_one_folder_up(tmp_path):
    root = _make_repo(tmp_path, notice=True, extra={"about.md": "built at Some Private Corp last year\n"})
    (tmp_path / ".prepublish-private").write_text("some private corp\n", encoding="utf-8")  # next to the repo folder
    r = _prepublish(root)
    assert r.returncode == 1 and "about.md" in r.stdout

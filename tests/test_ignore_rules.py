import subprocess


def ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path]).returncode == 0


def test_run_results_are_never_committed_here():
    """results.jsonl of a run can quote SQuAD (CC BY-SA) in model answers or judge reasoning."""
    assert ignored("runs/2026-01-01T00-00-00Z-anything/results.jsonl")
    assert ignored("runs/x-squad-y/results.jsonl")
    assert not ignored("runs/x/summary.json") and not ignored("runs/x/run_log.md")


def test_the_labelling_sheet_and_the_squad_download_are_ignored():
    assert ignored("calibration/labels.csv") and ignored("calibration/work/dev-v2.0.json")
    assert not ignored("calibration/labels_public.csv") and not ignored("calibration/question_ids.txt")

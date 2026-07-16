from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_outputs_from_mech.sh"


def run_sync(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake_rsync = bin_dir / "rsync"
    fake_rsync.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake_rsync.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--local-dir",
            str(tmp_path / "outputs"),
            *arguments,
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_default_excludes_weights_and_geometry(tmp_path: Path):
    result = run_sync(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--exclude=geometry/" in result.stdout
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        assert f"--exclude={pattern}" in result.stdout


def test_include_flags_are_independent(tmp_path: Path):
    with_pt = run_sync(tmp_path / "pt", "--include-pt")
    assert with_pt.returncode == 0, with_pt.stderr
    assert "--exclude=geometry/" in with_pt.stdout
    assert "--exclude=*.pt" not in with_pt.stdout

    with_geometry = run_sync(tmp_path / "geometry", "--include-geometry")
    assert with_geometry.returncode == 0, with_geometry.stderr
    assert "--exclude=geometry/" not in with_geometry.stdout
    assert "--exclude=*.pt" in with_geometry.stdout

    with_both = run_sync(
        tmp_path / "both", "--include-pt", "--include-geometry"
    )
    assert with_both.returncode == 0, with_both.stderr
    assert "--exclude=geometry/" not in with_both.stdout
    assert "--exclude=*.pt" not in with_both.stdout


def test_best_only_includes_best_and_excludes_other_weights(tmp_path: Path):
    result = run_sync(tmp_path, "--best-only")
    assert result.returncode == 0, result.stderr
    assert "--include=best.pt" in result.stdout
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        assert f"--exclude={pattern}" in result.stdout


def test_conflicting_model_flags_fail(tmp_path: Path):
    result = run_sync(tmp_path, "--include-pt", "--best-only")
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr

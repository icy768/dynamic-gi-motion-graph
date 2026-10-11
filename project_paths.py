"""All runtime resources are resolved from this project directory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TRANSFORMER_PATH = CHECKPOINT_DIR / "best_transformer.pt"
MOTION_PATH = CHECKPOINT_DIR / "best_spatial_motion_net.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_RUN = RUNS_DIR / "mnist_block3"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Runtime path must stay inside the project: {path}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict:
    digest = hashlib.sha256()
    sources = list(PROJECT_ROOT.glob("*.py"))
    for directory in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs"):
        if directory.exists():
            sources.extend(path for path in directory.rglob("*") if path.suffix in {".py", ".sh", ".ps1", ".json"})
    for path in sorted(sources):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    source_hash = digest.hexdigest()
    exported = PROJECT_ROOT / "CODE_VERSION.json"
    if exported.is_file():
        value = json.loads(exported.read_text(encoding="utf-8"))
        value["runtime_source_sha256"] = source_hash
        return value
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "working_tree_dirty": dirty, "runtime_source_sha256": source_hash}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None, "runtime_source_sha256": source_hash}


def asset_hashes() -> dict:
    return {
        "transformer": sha256_file(TRANSFORMER_PATH),
        "motion_estimator": sha256_file(MOTION_PATH),
    }

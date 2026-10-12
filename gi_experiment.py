"""Shared asset, configuration and frozen-model loading for the GI experiment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from project_paths import (
    PROJECT_ROOT, TRANSFORMER_PATH, MOTION_PATH, asset_hashes,
    code_version, project_path, sha256_file,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_config(path="configs/mnist_block3.json") -> dict:
    config = json.loads(project_path(path).read_text(encoding="utf-8"))
    expected = {"pattern_counts": [128, 256], "variants": ["cnn", "fixed", "motion"]}
    if config.get("protocol_version", 1) == 1:
        expected.update(train_sequences=800, validation_sequences=120, test_sequences=120)
    elif config["protocol_version"] != 2:
        raise ValueError("Unsupported experiment protocol")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"This protocol requires {key}={value}")
    for key in ("epochs", "bootstrap_repeats", "train_sequences", "validation_sequences", "test_sequences", "test_block"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "test_block" else 1):
            raise ValueError(f"Invalid {key}")
    if config["bootstrap_repeats"] < 100 or not 0 <= config["minimum_psnr_gain_db"] < 100:
        raise ValueError("epochs and bootstrap repeats must be positive")
    offsets = config["seed_offsets"]
    if len(offsets) < 3 or len(set(offsets)) != len(offsets) or any(type(x) is not int or x < 0 for x in offsets):
        raise ValueError("Use at least three distinct nonnegative seed offsets")
    if config["test_block"] < 3 or ("split_manifest" not in config and 120 + 120 * (config["test_block"] + 1) > 10000):
        raise ValueError("The final test must use a valid reserved block numbered 3 or later")
    if config.get("protocol_version", 1) == 2:
        if "split_manifest" not in config:
            raise ValueError("Expanded experiments require a prepared split manifest")
        from experiment_protocol import split_manifest
        split_manifest(config)
    project_path(config["run_directory"])
    return config


def choose_device(name="auto"):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def verify_assets() -> dict:
    manifest = json.loads((PROJECT_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
    files = manifest["required_files"]
    required = {f"data/MNIST/raw/{name}" for name in (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    )} | {"checkpoints/best_transformer.pt", "checkpoints/best_spatial_motion_net.pt"}
    if {item["path"] for item in files} != required:
        raise ValueError("Asset manifest does not contain the expected MNIST and frozen models")
    for item in files:
        path = project_path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Missing or changed asset: {item['path']}")
    package_count = 0
    package_manifest = PROJECT_ROOT / "PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        packaged = json.loads(package_manifest.read_text(encoding="utf-8"))
        for item in packaged["files"]:
            path = project_path(item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Packaged file changed: {item['path']}; prepare a new upload package")
        package_count = len(packaged["files"])
    return {"verified_files": len(files), "bytes": sum(item["bytes"] for item in files),
            "package_files_verified": package_count, "checkpoint_sha256": asset_hashes()}


def load_frozen(device):
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
    from train_spatial_motion_diagnostic import SpatialMotionNet
    transformer = MiniSpatioTemporalTransformer().to(device)
    motion = SpatialMotionNet().to(device)
    transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device, weights_only=True))
    motion.load_state_dict(torch.load(MOTION_PATH, map_location=device, weights_only=True))
    for model in (transformer, motion):
        model.eval()
        model.requires_grad_(False)
    return transformer, motion


def make_bank(device):
    from mini_repro_dynghost import make_patterns
    from train_graph_protocol_v2 import IMAGE_SIZE, RATES, SEED
    return {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}


def runtime_metadata(device) -> dict:
    names = ("torch", "torchvision", "numpy", "matplotlib", "pytorch-msssim", "Pillow")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_version": code_version(), "checkpoint_sha256": asset_hashes(),
    }

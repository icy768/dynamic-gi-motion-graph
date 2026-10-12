"""Compute shared frozen training features once; final test is never cached here."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time

import torch
from torch.utils.data import DataLoader

from experiment_protocol import protocol_signature, training_validation
from gi_experiment import atomic_torch_save, choose_device, load_config, load_frozen, make_bank, runtime_metadata, write_json
from project_paths import asset_hashes, code_version, project_path, sha256_file


def cache_signature(config, device):
    return {"protocol": protocol_signature(config), "frozen": asset_hashes(),
            "runtime_source_sha256": code_version()["runtime_source_sha256"],
            "packages": runtime_metadata(device)["packages"], "device": str(device),
            "gpu": runtime_metadata(device)["gpu"]}


def cache_paths(config):
    root = project_path(config["run_directory"]) / "feature_cache"
    return root, root / "metadata.json"


def cache_metadata(config, device):
    root, path = cache_paths(config)
    if not path.is_file():
        raise RuntimeError("Training cache missing; run prepare_experiment_cache.py first")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata["signature"] != cache_signature(config, device):
        raise RuntimeError("Training cache differs in data, frozen models, source or runtime; use a new run directory")
    for name in ("train", "val"):
        item = metadata["files"].get(name)
        if not item or not (root / item["name"]).is_file() or sha256_file(root / item["name"]) != item["sha256"]:
            raise RuntimeError(f"Training cache incomplete or damaged: {name}")
    return metadata


def ensure_cache(config, device):
    from train_madygraph_gi_plugin import precompute
    root, path = cache_paths(config)
    signature = cache_signature(config, device)
    metadata = {"signature": signature, "files": {}, "final_test_metrics_evaluated": False}
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["signature"] != signature:
            raise RuntimeError("Existing cache belongs to another experiment/runtime")
    train, validation = training_validation(config)
    transformer, motion = load_frozen(device)
    bank = make_bank(device)
    for name, dataset in (("train", train), ("val", validation)):
        file = root / f"{name}.pt"
        entry = metadata["files"].get(name)
        if entry and file.exists() and sha256_file(file) == entry["sha256"]:
            print(f"shared cache already complete: {name}", flush=True)
            continue
        start = time.perf_counter()
        data = precompute(DataLoader(dataset, batch_size=8, shuffle=False), transformer, motion, bank, device)
        atomic_torch_save(data, file)
        metadata["files"][name] = {"name": file.name, "bytes": file.stat().st_size,
                                   "sha256": sha256_file(file), "sequences": len(dataset),
                                   "elapsed_seconds": time.perf_counter() - start}
        write_json(path, metadata)
        del data
        gc.collect()
        print(f"shared cache saved: {name}, sequences={len(dataset)}", flush=True)
    cache_metadata(config, device)


def load_cache(config, device):
    metadata = cache_metadata(config, device)
    root, _ = cache_paths(config)
    data = {name: torch.load(root / entry["name"], map_location="cpu", weights_only=True)
            for name, entry in metadata["files"].items()}
    for name, count in (("train", config["train_sequences"]), ("val", config["validation_sequences"])):
        for rate in config["pattern_counts"]:
            if len(data[name][rate]["target"]) != count:
                raise ValueError("Cached training count differs")
    return data, hashlib.sha256(json.dumps(metadata["signature"], sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_scaleup.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    ensure_cache(load_config(args.config), choose_device(args.device))

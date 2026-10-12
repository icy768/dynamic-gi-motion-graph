"""Fixed identities and trajectories for legacy and expanded GI protocols."""
from __future__ import annotations

import json

import numpy as np

from project_paths import PROJECT_ROOT, project_path, sha256_file
from train_graph_protocol_v2 import BouncingMovingMNIST, SEED, make_test_block, make_train_validation


def split_manifest(config):
    if "split_manifest" not in config:
        return None
    path = project_path(config["split_manifest"])
    if sha256_file(path) != config.get("split_manifest_sha256"):
        raise ValueError("Prepared split manifest changed; use a new experiment configuration")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest schema")
    excluded = set(value["historically_exposed_test_indices"])
    for name, total, count_key in (("train", 60000, "train_sequences"),
                                   ("validation", 10000, "validation_sequences"),
                                   ("test", 10000, "test_sequences")):
        item = value["splits"][name]
        ids = item["mnist_indices"]
        if len(ids) != config[count_key] or len(set(ids)) != len(ids):
            raise ValueError(f"Wrong count or duplicate identities in {name}")
        if any(type(i) is not int or not 0 <= i < total for i in ids):
            raise ValueError(f"Invalid MNIST identity in {name}")
        if item["source"] != ("official_train" if name == "train" else "official_test"):
            raise ValueError(f"Wrong MNIST source for {name}")
        if type(item["trajectory_seed"]) is not int:
            raise ValueError("Invalid trajectory seed")
        if name != "train" and set(ids) & excluded:
            raise ValueError(f"{name} contains historically exposed MNIST identities")
    if set(value["splits"]["validation"]["mnist_indices"]) & set(value["splits"]["test"]["mnist_indices"]):
        raise ValueError("Validation and final test identities overlap")
    return value


def protocol_signature(config):
    return {"split_manifest_sha256": config.get("split_manifest_sha256"),
            "counts": [config[k] for k in ("train_sequences", "validation_sequences", "test_sequences")],
            "test_block": config["test_block"]}


def _dataset(item):
    return BouncingMovingMNIST(PROJECT_ROOT / "data", item["source"] == "official_train",
                               np.asarray(item["mnist_indices"]), item["trajectory_seed"])


def training_validation(config):
    prepared = split_manifest(config)
    if prepared:
        return _dataset(prepared["splits"]["train"]), _dataset(prepared["splits"]["validation"])
    return make_train_validation(PROJECT_ROOT)


def final_test(config):
    prepared = split_manifest(config)
    return _dataset(prepared["splits"]["test"]) if prepared else make_test_block(PROJECT_ROOT, config["test_block"])


def protocol_report(config):
    prepared = split_manifest(config)
    return {"protocol": protocol_signature(config),
            "historical_exposure_exclusions": len(prepared["historically_exposed_test_indices"]) if prepared else None,
            "validation_final_identity_overlap": 0,
            "final_test_images_generated": False, "final_test_metrics_evaluated": False,
            "dataset": "Single-digit bouncing Moving-MNIST; 8 frames, 64x64, speed 1-6 px/frame",
            "raw_data_download_required": False}

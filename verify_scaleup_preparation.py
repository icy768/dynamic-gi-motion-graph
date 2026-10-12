"""Check the enlarged pipeline using development identities, never its reserved test."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

import torch

from experiment_protocol import protocol_report, split_manifest
from gi_experiment import atomic_torch_save, choose_device, load_config, write_json
from package_gi_project import result_package
from prepare_experiment_cache import load_cache
from project_paths import PROJECT_ROOT, code_version, sha256_file
from train_graph_protocol_v2 import SEED, split_indices
from train_madygraph_gi_plugin import train_one


def verify(device):
    production = load_config("configs/mnist_scaleup.json")
    prepared = split_manifest(production)
    directory = PROJECT_ROOT / "tmp" / "scaleup_checks" / str(time.time_ns())
    directory.mkdir(parents=True)
    _, old_perm = split_indices()
    development = copy.deepcopy(prepared)
    development["study_id"] = "development_check"
    development["historically_exposed_test_indices"] = []
    development["splits"]["train"]["mnist_indices"] = development["splits"]["train"]["mnist_indices"][:16]
    # Reuse already exposed legacy validation identities ONLY for engineering checks.
    development["splits"]["validation"]["mnist_indices"] = old_perm[:8].tolist()
    development["splits"]["test"]["mnist_indices"] = old_perm[8:16].tolist()
    reserved = set(prepared["splits"]["test"]["mnist_indices"])
    assert reserved.isdisjoint(old_perm[:16].tolist())
    manifest = directory / "development_splits.json"
    write_json(manifest, development)
    config = copy.deepcopy(production)
    config.update(study_id="development_check", train_sequences=16, validation_sequences=8, test_sequences=8,
                  epochs=2, seed_offsets=[0, 1, 2], run_directory=(directory / "run").relative_to(PROJECT_ROOT).as_posix(),
                  split_manifest=manifest.relative_to(PROJECT_ROOT).as_posix(), split_manifest_sha256=sha256_file(manifest),
                  bootstrap_repeats=100)
    config_file = directory / "development_config.json"
    write_json(config_file, config)
    subprocess.run([sys.executable, "-u", "run_server_experiment.py", "--config", str(config_file),
                    "--stage", "all", "--device", str(device)], cwd=PROJECT_ROOT, check=True)
    data, _ = load_cache(config, device)
    for rate in production["pattern_counts"]:
        scale = max(float(data["train"][rate]["residual"].square().mean().sqrt()), 1e-8)
        for name in data:
            data[name][rate]["residual"] = (data[name][rate]["residual"] / scale).clamp(-8, 8)
    uninterrupted, resumed = directory / "uninterrupted", directory / "resumed"
    uninterrupted.mkdir()
    resumed.mkdir()
    first, full_history, _ = train_one("motion", data, device, 2, uninterrupted, SEED + 910, "resume_check")
    expected = {key: value.detach().cpu().clone() for key, value in first.state_dict().items()}
    del first

    def simulate_interruption(value, path):
        atomic_torch_save(value, path)
        if path.name == "last_motion_training.pt" and value["epoch"] == 1:
            raise InterruptedError("Simulated interruption after an atomically saved epoch")

    try:
        with patch("train_madygraph_gi_plugin.atomic_torch_save", simulate_interruption):
            train_one("motion", data, device, 2, resumed, SEED + 910, "resume_check")
    except InterruptedError:
        pass
    recovered, recovered_history, _ = train_one("motion", data, device, 2, resumed, SEED + 910, "resume_check")
    largest = max(float((expected[key] - value.detach().cpu()).abs().max()) for key, value in recovered.state_dict().items())
    if largest > 1e-6:
        raise AssertionError(f"Resumed weights differ: {largest}")
    history_difference = 0.0
    for left, right in zip(full_history, recovered_history):
        for key in ("train_loss", "val_mse", "val_ssim", "val_selection_score", "minibatch_order_sha256"):
            if key != "minibatch_order_sha256":
                history_difference = max(history_difference, abs(left[key] - right[key]))
                matches = abs(left[key] - right[key]) <= 1e-7
            else:
                matches = left[key] == right[key]
            if not matches:
                raise AssertionError(f"Resumed training differs: {key}")
    from evaluate_madygraph_gi_final import evaluate_run
    changed = copy.deepcopy(config)
    changed["minimum_psnr_gain_db"] += 0.1
    try:
        evaluate_run(changed, directory / "run", device)
    except RuntimeError as error:
        if "changed" not in str(error):
            raise
    else:
        raise AssertionError("Changed final-test configuration was accepted")
    archive = result_package(str(config_file), str((directory / "development_results.zip").relative_to(PROJECT_ROOT)))
    with zipfile.ZipFile(archive) as packed:
        assert packed.testzip() is None
        assert not any(name.endswith("/train.pt") or name.endswith("/val.pt") for name in packed.namelist())
        assert "gi_results/prepared_split_manifest.json" in packed.namelist()
    report = {"passed": True, "runtime_source_sha256": code_version()["runtime_source_sha256"],
              "production_protocol": protocol_report(production), "device": str(device),
              "checks": ["assets and split manifest", "historical identity exclusions", "3-seed development train/evaluate/benchmark",
                         "variant order invariance", "shared cache", "epoch resume equivalence", "final-test change guard", "result ZIP integrity"],
              "resume_max_parameter_difference": largest,
              "resume_max_metric_difference": history_difference,
              "development_final_identities": old_perm[8:16].tolist(),
              "production_final_test_images_generated": False, "production_final_test_metrics_evaluated": False,
              "development_directory": directory.relative_to(PROJECT_ROOT).as_posix()}
    write_json(PROJECT_ROOT / "SCALEUP_PREPARATION_CHECKS.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    verify(choose_device(args.device))

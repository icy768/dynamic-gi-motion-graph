"""One project-root entry point for preparation and the reserved MNIST experiment."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from gi_experiment import (
    choose_device, load_config, load_frozen, make_bank, runtime_metadata, verify_assets, write_json,
)
from project_paths import PROJECT_ROOT, project_path


def command(script, *args):
    subprocess.run([sys.executable, str(PROJECT_ROOT / script), *map(str, args)], cwd=PROJECT_ROOT, check=True)


def preflight(device):
    assets = verify_assets()
    syntax_count = 0
    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if relative.parts[0] in {"tmp", ".tmp_paper_notes", ".local-tools", ".local-backup", "dist", "runs"}:
            continue
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(relative))
        if "D:\\download\\DynGhost-mini-repro" in source:
            raise RuntimeError(f"External runtime path remains: {relative}")
        syntax_count += 1
    from train_graph_protocol_v2 import IMAGE_SIZE, make_train_validation
    from train_madygraph_gi_plugin import ResidualRefiner, dgi_from_bucket, gi_forward, physical_residual, precompute
    from train_mixed_sampling import coarse_sequence
    _, validation = make_train_validation(PROJECT_ROOT)
    transformer, motion = load_frozen(device)
    bank = make_bank(device)
    loader = DataLoader(Subset(validation, range(2)), batch_size=2, shuffle=False)
    data = precompute(loader, transformer, motion, bank, device)
    checks = []
    with torch.inference_mode():
        for rate, item in data.items():
            target = item["target"].to(device)
            patterns = bank[rate]
            bucket = gi_forward(target, patterns)
            actual = dgi_from_bucket(bucket, patterns)
            expected = coarse_sequence(target, patterns)
            dgi_error = float((actual - expected).abs().max())
            if dgi_error > 2e-5:
                raise RuntimeError(f"Bucket-driven DGI differs from the reference: {dgi_error}")
            zero = physical_residual(bucket, target, patterns)
            if float(zero.abs().max()) > 1e-7:
                raise RuntimeError("Measurement residual does not vanish for the measured image")
            for kind in ("cnn", "fixed", "motion"):
                model = ResidualRefiner(kind).to(device).eval()
                prediction = model(item["coarse"].to(device), item["residual"].to(device), item["motion"].to(device))
                if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                    raise RuntimeError(f"Invalid {kind} forward output")
                del model
            checks.append({"pattern_count": rate, "bucket_dgi_max_abs_difference": dgi_error,
                           "zero_measurement_residual_max_abs": float(zero.abs().max()),
                           "three_refiner_forward_shapes": list(target.shape)})
    report = {"passed": True, "assets": assets, "python_files_parsed": syntax_count,
              "runtime": runtime_metadata(device), "validation_sequences_used": 2,
              "reserved_final_test_read": False, "checks": checks}
    write_json(PROJECT_ROOT / "runs" / "preflight_check.json", report)
    print(f"preflight passed: {assets['verified_files']} assets, {syntax_count} Python modules, device={device}", flush=True)
    return report


def smoke(config, device):
    directory = PROJECT_ROOT / "runs" / "preparation_smoke"
    if (directory / "checks.json").is_file():
        existing = json.loads((directory / "checks.json").read_text(encoding="utf-8"))
        current = runtime_metadata(device)
        matching = all(existing["runtime"].get(key) == current[key]
                       for key in ("code_version", "device", "gpu", "checkpoint_sha256", "packages"))
        if existing.get("passed") and matching:
            print(f"smoke checks already passed: {directory}", flush=True)
            return existing
    token = str(time.time_ns())
    directory = directory / token
    common = ["--epochs", 1, "--seed-offset", 0, "--skip-test", "--train-limit", 16, "--val-limit", 8, "--device", str(device)]
    command("train_madygraph_gi_plugin.py", *common, "--variants", "cnn,fixed,motion", "--output-dir", directory / "seed0")
    command("train_madygraph_gi_plugin.py", *common, "--variants", "motion,fixed,cnn", "--output-dir", directory / "order_check")
    largest_difference = 0.0
    for kind in config["variants"]:
        left = torch.load(directory / "seed0" / f"best_{kind}_refiner.pt", map_location="cpu", weights_only=True)
        right = torch.load(directory / "order_check" / f"best_{kind}_refiner.pt", map_location="cpu", weights_only=True)
        for key in left:
            difference = float((left[key] - right[key]).abs().max())
            largest_difference = max(largest_difference, difference)
            if not torch.allclose(left[key], right[key], rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"Variant ordering changed {kind}/{key}: {difference}")
    from evaluate_madygraph_gi_final import evaluate_run
    evaluate_run(config, directory, device, validation_only=True, sample_limit=8, offsets=[0])
    from benchmark_madygraph_gi_plugin_efficiency import benchmark
    benchmark(directory / "seed0", directory / "efficiency.json", device, warmup=2, repeats=5)
    report = {"passed": True, "runtime": runtime_metadata(device),
              "training_sequences": 16, "validation_sequences": 8, "epochs": 1,
              "variant_order_max_parameter_difference": largest_difference,
              "reserved_final_test_read": False, "directory": directory.relative_to(PROJECT_ROOT).as_posix(),
              "completed": ["forward", "backward", "checkpoint_selection", "variant_order_invariance",
                            "validation_metrics", "paired_statistics", "figures", "stage_timing"]}
    write_json(directory / "checks.json", report)
    write_json(PROJECT_ROOT / "runs" / "preparation_smoke" / "checks.json", report)
    print(f"smoke passed; final block remains reserved: {directory}", flush=True)
    return report


def train(config, device):
    root = project_path(config["run_directory"])
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "experiment_config.json"
    if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Configuration changed inside an existing experiment directory")
    write_json(snapshot, config)
    for offset in config["seed_offsets"]:
        directory = root / f"seed{offset}"
        if (directory / "results.json").exists():
            from evaluate_madygraph_gi_final import validate_training
            validate_training(root, config, [offset])
            print(f"training already complete: seed{offset}", flush=True)
            continue
        command("train_madygraph_gi_plugin.py", "--epochs", config["epochs"], "--variants", ",".join(config["variants"]),
                "--seed-offset", offset, "--skip-test", "--test-block", config["test_block"],
                "--output-dir", directory, "--device", str(device))


def main():
    parser = argparse.ArgumentParser(description="Run all resources from the directory containing this script")
    parser.add_argument("--stage", choices=("check", "smoke", "train", "evaluate", "benchmark", "all"), default="check")
    parser.add_argument("--config", default="configs/mnist_block3.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device("cuda" if args.require_cuda else args.device)
    started = time.perf_counter()
    preflight(device)
    if args.stage in {"smoke", "all"}:
        smoke(config, device)
    if args.stage in {"train", "all"}:
        train(config, device)
    if args.stage in {"evaluate", "all"}:
        from evaluate_madygraph_gi_final import evaluate_run
        evaluate_run(config, project_path(config["run_directory"]), device)
    if args.stage in {"benchmark", "all"}:
        from benchmark_madygraph_gi_plugin_efficiency import benchmark
        run = project_path(config["run_directory"])
        benchmark(run / f"seed{config['seed_offsets'][0]}", run / "efficiency.json", device)
    print(f"stage={args.stage} completed in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()

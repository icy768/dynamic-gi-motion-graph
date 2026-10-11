"""Evaluate frozen refiners once on the reserved block; bootstrap paired identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gi_experiment import (
    choose_device, load_config, load_frozen, make_bank, runtime_metadata, verify_assets, write_json,
)
from project_paths import PROJECT_ROOT, asset_hashes, code_version, project_path, sha256_file
from train_graph_protocol_v2 import (
    IMAGE_SIZE, RATES, SEED, make_test_block, make_train_validation, seed_all, speed_bin_name,
)
from train_madygraph_gi_plugin import (
    ResidualRefiner, csv_write, metric_rows, precompute, relative_measurement_residual, summarize,
)

METRICS = ("mse", "psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error", "relative_measurement_residual")
LABELS = {"coarse": "Coarse", "cnn": "CNN", "fixed": "Fixed", "motion": "Motion"}


def validate_training(run_dir, config, offsets, diagnostic=False):
    manifests, signature = {}, []
    expected_hashes = asset_hashes()
    for offset in offsets:
        directory = run_dir / f"seed{offset}"
        results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        manifest = results["manifest"]
        if manifest["phase"] != "training_only" or manifest["test_evaluated"]:
            raise ValueError("Final evaluation requires training-only checkpoints")
        if manifest["seed_offset"] != offset or manifest["checkpoint_sha256"] != expected_hashes:
            raise ValueError("Seed or frozen checkpoint identity differs")
        if manifest["runtime"]["code_version"].get("runtime_source_sha256") != code_version()["runtime_source_sha256"]:
            raise ValueError("Runtime source changed since training; use the original experiment version")
        if manifest["rates"] != config["pattern_counts"] or set(manifest["graph_variants"]) != set(config["variants"]):
            raise ValueError("Training variants or measurement rates differ")
        if not diagnostic and (manifest["epochs"] != config["epochs"] or manifest["train_val_test_sequences"] != [800, 120, 0]
                               or manifest["reserved_test_block"] != config["test_block"]):
            raise ValueError("Training budget or reserved block differs from the final protocol")
        orders = [[epoch["minibatch_order_sha256"] for epoch in results["histories"][kind]] for kind in config["variants"]]
        if any(order != orders[0] for order in orders[1:]):
            raise ValueError("Variant batch orders are not paired")
        item = {"offset": offset, "manifest_sha256": sha256_file(directory / "data_manifest.json"), "models": {}}
        for kind in config["variants"]:
            item["models"][kind] = sha256_file(directory / f"best_{kind}_refiner.pt")
        signature.append(item)
        manifests[offset] = manifest
    if not diagnostic:
        reference = manifests[offsets[0]]
        for manifest in manifests.values():
            for key in ("training_mnist_indices", "validation_mnist_indices"):
                if manifest[key] != reference[key]:
                    raise ValueError("Training and validation identities differ across seeds")
    return manifests, signature


def averaged_rows(rows, offsets):
    groups = {}
    for row in rows:
        groups.setdefault((row["pattern_count"], row["method"], row["mnist_index"]), []).append(row)
    output = []
    for (rate, method, identity), group in sorted(groups.items()):
        expected = 1 if method == "coarse" else len(offsets)
        if len(group) != expected:
            raise ValueError("Incomplete sequence/seed coverage")
        base = {key: group[0][key] for key in ("label", "speed_bin", "speed_pixels_per_frame")}
        output.append({"pattern_count": rate, "method": method, "mnist_index": identity, **base,
                       **{metric: float(np.mean([row[metric] for row in group])) for metric in METRICS}})
    return output


def paired_statistics(rows, mean_rows, offsets, repeats=5000):
    lookup = {(row["pattern_count"], row["method"], row["mnist_index"]): row for row in mean_rows}
    by_seed = {(row["pattern_count"], row["method"], row["seed_offset"], row["mnist_index"]): row for row in rows}
    paired, seed_results = [], []
    contrasts = (("motion", "cnn"), ("motion", "fixed"), ("motion", "coarse"), ("cnn", "coarse"), ("fixed", "coarse"))
    for rate in RATES:
        coarse_rows = [row for row in mean_rows if row["pattern_count"] == rate and row["method"] == "coarse"]
        for speed_bin in ("1-2", "2-4", "4-6", "all"):
            ids = [row["mnist_index"] for row in coarse_rows if speed_bin == "all" or row["speed_bin"] == speed_bin]
            if not ids:
                continue
            for treatment, control in contrasts:
                a = [lookup[(rate, treatment, identity)] for identity in ids]
                b = [lookup[(rate, control, identity)] for identity in ids]
                delta = np.array([x["psnr_db"] - y["psnr_db"] for x, y in zip(a, b)])
                rng = np.random.default_rng(SEED + rate + 500)
                choices = rng.integers(0, len(ids), size=(repeats, len(ids)))
                distribution = delta[choices].mean(1)
                lower, upper = np.percentile(distribution, [2.5, 97.5])
                seed_means = []
                for offset in offsets:
                    seed_delta = []
                    for identity in ids:
                        x = by_seed[(rate, treatment, offset, identity)]
                        y = by_seed[(rate, control, -1 if control == "coarse" else offset, identity)]
                        seed_delta.append(x["psnr_db"] - y["psnr_db"])
                    seed_means.append(float(np.mean(seed_delta)))
                    seed_results.append({"pattern_count": rate, "speed_bin": speed_bin,
                                         "comparison": f"{treatment}_vs_{control}", "seed_offset": offset,
                                         "paired_psnr_gain_db": seed_means[-1]})
                control_mse = np.mean([row["mse"] for row in b])
                paired.append({
                    "pattern_count": rate, "speed_bin": speed_bin, "comparison": f"{treatment}_vs_{control}",
                    "sequences": len(ids), "refiner_seeds": len(offsets),
                    "paired_psnr_gain_mean_db": float(delta.mean()),
                    "psnr_gain_ci95_lower_db": float(lower), "psnr_gain_ci95_upper_db": float(upper),
                    "refiner_seed_gain_std_db": float(np.std(seed_means, ddof=1)) if len(offsets) > 1 else 0.0,
                    "paired_win_rate": float(np.mean(delta > 0)),
                    "ssim_gain_mean": float(np.mean([x["ssim"] - y["ssim"] for x, y in zip(a, b)])),
                    "mse_reduction_percent": float(100 * (control_mse - np.mean([row["mse"] for row in a])) / max(control_mse, 1e-12)),
                    "temporal_error_reduction_percent": float(100 * (np.mean([row["temporal_difference_error"] for row in b]) - np.mean([row["temporal_difference_error"] for row in a])) / max(np.mean([row["temporal_difference_error"] for row in b]), 1e-12)),
                    "measurement_residual_change": float(np.mean([x["relative_measurement_residual"] - y["relative_measurement_residual"] for x, y in zip(a, b)])),
                })
    return paired, seed_results


def save_figures(output, summary, paired, examples):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    methods = ("coarse", "cnn", "fixed", "motion")
    colors = ("#88939d", "#4477aa", "#66aabb", "#cc6677")
    overall = {(row["pattern_count"], row["method"]): row for row in summary if row["speed_bin"] == "all"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ("psnr_db", "ssim"), ("PSNR (dB)", "SSIM")):
        for index, (method, color) in enumerate(zip(methods, colors)):
            ax.bar(np.arange(2) + (index - 1.5) * .18,
                   [overall[(rate, method)][metric + "_mean"] for rate in RATES],
                   width=.18, color=color, label=LABELS[method])
        ax.set_xticks([0, 1], [str(rate) for rate in RATES])
        ax.set_xlabel("Patterns")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, control in enumerate(("cnn", "fixed", "coarse")):
        entries = [next(row for row in paired if row["pattern_count"] == rate and row["speed_bin"] == "all" and row["comparison"] == "motion_vs_" + control) for rate in RATES]
        means = np.array([row["paired_psnr_gain_mean_db"] for row in entries])
        lower = np.array([row["psnr_gain_ci95_lower_db"] for row in entries])
        upper = np.array([row["psnr_gain_ci95_upper_db"] for row in entries])
        ax.errorbar(np.arange(2) + (index - 1) * .12, means, yerr=np.maximum(0, np.stack((means - lower, upper - means))),
                    marker="o", capsize=4, label="vs " + LABELS[control])
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set_xticks([0, 1], [str(rate) for rate in RATES])
    ax.set_xlabel("Patterns")
    ax.set_ylabel("Motion PSNR gain (dB)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "paired_gains.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, (method, color) in enumerate(zip(methods, colors)):
        ax.bar(np.arange(2) + (index - 1.5) * .18,
               [overall[(rate, method)]["relative_measurement_residual_mean"] for rate in RATES],
               width=.18, color=color, label=LABELS[method])
    ax.set_xticks([0, 1], [str(rate) for rate in RATES])
    ax.set_xlabel("Patterns")
    ax.set_ylabel("Relative measurement residual")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "measurement_consistency.png", dpi=180)
    plt.close(fig)
    if examples:
        sample_ids = sorted(examples)
        fig, axes = plt.subplots(5, len(sample_ids) * 2, figsize=(3 * len(sample_ids), 7), squeeze=False)
        for col, identity in enumerate(sample_ids):
            item = examples[identity]
            for row, method in enumerate(("target", *methods)):
                for frame_index, frame in enumerate((0, 7)):
                    ax = axes[row, 2 * col + frame_index]
                    ax.imshow(item[method][frame, 0], cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0 and frame_index == 0:
                        ax.set_ylabel("Target" if method == "target" else LABELS[method])
                    if row == 0:
                        ax.set_title(f"{item['speed_bin']}  t{frame}", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / "example_reconstructions.png", dpi=180)
        plt.close(fig)


def evaluate_run(config, run_dir, device, validation_only=False, sample_limit=8, offsets=None):
    offsets = config["seed_offsets"] if offsets is None else offsets
    if not validation_only and offsets != config["seed_offsets"]:
        raise ValueError("Final evaluation requires all configured seeds")
    manifests, training_signature = validate_training(run_dir, config, offsets, diagnostic=validation_only)
    output = run_dir / ("validation_diagnostic" if validation_only else "final")
    fingerprint_payload = {"config": config, "training": training_signature, "frozen": asset_hashes()}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    if not validation_only:
        guard = run_dir / "final_evaluation_guard.json"
        if guard.exists():
            state = json.loads(guard.read_text(encoding="utf-8"))
            if state["fingerprint"] != fingerprint:
                raise RuntimeError("Final-test checkpoints/config changed; reserve a new test block for another experiment")
            if (output / "results.json").is_file():
                print(f"final evaluation already complete: {output}", flush=True)
                return json.loads((output / "results.json").read_text(encoding="utf-8"))
        write_json(guard, {"fingerprint": fingerprint, "test_block": config["test_block"],
                           "training_signature": training_signature, "runtime": runtime_metadata(device)})
    output.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    if validation_only:
        _, validation = make_train_validation(PROJECT_ROOT)
        dataset = Subset(validation, range(min(sample_limit, len(validation))))
    else:
        dataset = make_test_block(PROJECT_ROOT, config["test_block"])
        if set(dataset.indices.tolist()) & set(manifests[offsets[0]]["validation_mnist_indices"]):
            raise RuntimeError("Final and validation MNIST identities overlap")
    transformer, motion_net = load_frozen(device)
    bank = make_bank(device)
    start = time.perf_counter()
    splits = precompute(DataLoader(dataset, batch_size=4, shuffle=False), transformer, motion_net, bank, device)
    rows, examples = [], {}
    chosen = {}
    first = splits[RATES[0]]
    for i, speed in enumerate(first["speed"]):
        chosen.setdefault(speed_bin_name(float(speed)), i)
    for index in chosen.values():
        identity = int(first["indices"][index])
        examples[identity] = {"target": first["target"][index].numpy(),
                              "coarse": first["coarse"][index].clamp(0, 1).numpy(),
                              "speed_bin": speed_bin_name(float(first["speed"][index]))}
    for rate, data in splits.items():
        prediction = data["coarse"].clamp(0, 1)
        residuals = relative_measurement_residual(prediction.to(device), data["bucket"].to(device), bank[rate])
        for i, metrics in enumerate(metric_rows(prediction, data["target"], data["labels"], data["indices"])):
            rows.append({"pattern_count": rate, "method": "coarse", "seed_offset": -1,
                         "speed_pixels_per_frame": float(data["speed"][i]), "speed_bin": speed_bin_name(float(data["speed"][i])),
                         **metrics, "relative_measurement_residual": float(residuals[i])})
    with torch.inference_mode():
        for offset in offsets:
            for kind in config["variants"]:
                model = ResidualRefiner(kind).to(device).eval()
                model.load_state_dict(torch.load(run_dir / f"seed{offset}" / f"best_{kind}_refiner.pt", map_location=device, weights_only=True))
                for rate, data in splits.items():
                    scale = manifests[offset]["measurement_residual_train_rms_by_rate"][str(rate)]
                    for begin in range(0, len(data["target"]), 4):
                        batch = {key: value[begin:begin + 4].to(device) for key, value in data.items()}
                        residual = (batch["residual"] / scale).clamp(-8, 8)
                        prediction = model(batch["coarse"], residual, batch["motion"]).clamp(0, 1)
                        consistency = relative_measurement_residual(prediction, batch["bucket"], bank[rate])
                        for i, metrics in enumerate(metric_rows(prediction, batch["target"], batch["labels"], batch["indices"])):
                            rows.append({"pattern_count": rate, "method": kind, "seed_offset": offset,
                                         "speed_pixels_per_frame": float(batch["speed"][i]), "speed_bin": speed_bin_name(float(batch["speed"][i])),
                                         **metrics, "relative_measurement_residual": float(consistency[i])})
                            if offset == offsets[0] and rate == RATES[0] and metrics["mnist_index"] in examples:
                                examples[metrics["mnist_index"]][kind] = prediction[i].cpu().numpy()
                del model
    mean_rows = averaged_rows(rows, offsets)
    summary = summarize(mean_rows)
    paired, seed_results = paired_statistics(rows, mean_rows, offsets, repeats=config["bootstrap_repeats"])
    checks = []
    for rate in RATES:
        row = next(item for item in paired if item["pattern_count"] == rate and item["speed_bin"] == "all" and item["comparison"] == "motion_vs_cnn")
        checks.append({"pattern_count": rate,
                       "gain_at_least_threshold": row["paired_psnr_gain_mean_db"] >= config["minimum_psnr_gain_db"],
                       "ci95_positive": row["psnr_gain_ci95_lower_db"] > 0, "ssim_not_lower": row["ssim_gain_mean"] >= 0})
    csv_write(output / "per_sequence_results.csv", rows)
    csv_write(output / "benchmark_summary.csv", summary)
    csv_write(output / "paired_gains.csv", paired)
    csv_write(output / "seed_gains.csv", seed_results)
    save_figures(output, summary, paired, examples)
    result = {
        "phase": "validation_diagnostic" if validation_only else "final_test",
        "config": config, "runtime": runtime_metadata(device), "training_signature": training_signature,
        "sequences": len(dataset), "seed_offsets": offsets, "test_block": None if validation_only else config["test_block"],
        "test_mnist_indices": [] if validation_only else dataset.indices.tolist(),
        "summary": summary, "paired_gains": paired, "acceptance_checks": checks,
        "acceptance_evaluated": not validation_only,
        "accepted": None if validation_only else all(all(check[key] for key in ("gain_at_least_threshold", "ci95_positive", "ssim_not_lower")) for check in checks),
        "confidence_interval_scope": "Paired bootstrap over MNIST identities after averaging each identity's metrics over refiner seeds; conditional on the shared frozen backbone and motion estimator. Seed variation is reported separately.",
        "mse_percentage_definition": "100*(mean_control_mse-mean_treatment_mse)/mean_control_mse; means over identities and refiner seeds, not a PSNR percentage",
        "metric_clipping": "All output images are clipped to [0,1] for image and measurement metrics",
        "measurement_residual_definition": "||A*x-y||_2 / ||y||_2 over all frames/buckets of a sequence",
        "example_selection": "First sequence in each speed stratum; 128 patterns, first configured refiner seed; frames 0 and 7",
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(output / "results.json", result)
    print(f"evaluation_phase={result['phase']} sequences={len(dataset)} saved={output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_block3.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=None)
    args = parser.parse_args()
    verify_assets()
    config = load_config(args.config)
    if args.sample_limit < 1:
        parser.error("sample limit must be positive")
    evaluate_run(config, project_path(args.run_dir or config["run_directory"]), choose_device(args.device),
                 validation_only=args.validation_only, sample_limit=args.sample_limit, offsets=args.seed_offsets)


if __name__ == "__main__":
    main()

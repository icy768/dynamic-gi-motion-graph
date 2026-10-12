"""Reserve expanded train/validation/test identities without evaluating test images."""
from __future__ import annotations

import csv
import json

import numpy as np

from gi_experiment import verify_assets, write_json
from project_paths import PROJECT_ROOT, sha256_file
from train_graph_protocol_v2 import SEED, split_indices


def historical_exposure():
    _, test_perm = split_indices()
    exposed = set(test_perm[:600].tolist())  # validation and previously used blocks 0-3
    sources = [{"source": "Fixed protocol: validation and test blocks 0-3", "unique_identities": len(exposed)}]
    # Earlier checkpoints used random identities rather than the later fixed permutation.
    cases = (("Frame CNN validation", SEED + 2, 300),
             ("ConvGRU validation", SEED + 11, 120),
             ("Frozen/legacy Transformer validation", SEED + 201, 120),
             ("Legacy Transformer final evaluation", SEED + 300, 100),
             ("Legacy benchmark evaluation", SEED + 100, 100))
    for name, seed, count in cases:
        ids = set(np.random.default_rng(seed).integers(0, 10000, size=count).tolist())
        exposed.update(ids)
        sources.append({"source": name, "reconstructed_identity_seed": seed,
                        "samples": count, "unique_identities": len(ids)})
    exposed.add(7)  # original illustrative reconstruction
    ignored = {".git", "data", "checkpoints", "dist", "tmp", "configs", ".local-backup", ".tmp_paper_notes", "__pycache__"}

    def collect(value):
        ids = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"training_mnist_indices", "training_indices"}:
                    continue
                if key in {"test_mnist_indices", "validation_mnist_indices"} and isinstance(child, list):
                    ids.update(i for i in child if type(i) is int and 0 <= i < 10000)
                elif key == "mnist_index" and type(child) is int and 0 <= child < 10000:
                    ids.add(child)
                else:
                    ids.update(collect(child))
        elif isinstance(value, list):
            for child in value:
                ids.update(collect(child))
        return ids

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.relative_to(PROJECT_ROOT).parts[0] in ignored:
            continue
        if path.suffix == ".json":
            ids = collect(json.loads(path.read_text(encoding="utf-8-sig")))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                ids = {int(row["mnist_index"]) for row in reader
                       if "mnist_index" in row and row["mnist_index"].strip().isdigit()
                       and 0 <= int(row["mnist_index"]) < 10000}
        else:
            continue
        if ids:
            exposed.update(ids)
            sources.append({"source": path.relative_to(PROJECT_ROOT).as_posix(),
                            "sha256": sha256_file(path), "unique_identities": len(ids)})
    return exposed, sources


def prepare():
    verify_assets()
    manifest_path = PROJECT_ROOT / "configs" / "mnist_scaleup_splits.json"
    config_path = PROJECT_ROOT / "configs" / "mnist_scaleup.json"
    if manifest_path.exists() or config_path.exists():
        from gi_experiment import load_config
        from experiment_protocol import protocol_report
        report = protocol_report(load_config(config_path))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    excluded, audit = historical_exposure()
    train_perm, test_perm = split_indices()
    available = [int(i) for i in test_perm[600:] if int(i) not in excluded]
    if len(available) < 1200:
        raise RuntimeError("Not enough unused official-test identities")
    splits = {
        "train": {"source": "official_train", "mnist_indices": train_perm[:4000].tolist(), "trajectory_seed": SEED + 10701},
        "validation": {"source": "official_test", "mnist_indices": available[:600], "trajectory_seed": SEED + 10702},
        "test": {"source": "official_test", "mnist_indices": available[600:1200], "trajectory_seed": SEED + 10707},
    }
    manifest = {"schema_version": 1, "study_id": "mnist_scaleup_v1", "data_seed": SEED,
                "selection": "Fixed permutation, exclude known historical exposures, take first 600 for validation and next 600 for final test",
                "historically_exposed_test_indices": sorted(excluded), "historical_exposure_sources": audit,
                "audit_scope": "Existing local result records plus reconstructed legacy validation/evaluation samplers; excludes known exposure, not a claim about unrecorded experiments",
                "splits": splits, "resolution": [64, 64], "frames": 8,
                "motion": "One digit, balanced speed strata 1-2/2-4/4-6 px/frame, reflection at +/-10 px",
                "raw_data": {item["path"]: item["sha256"] for item in json.loads((PROJECT_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))["required_files"] if item["path"].startswith("data/")},
                "final_test_images_generated": False, "final_test_metrics_evaluated": False}
    write_json(manifest_path, manifest)
    config = {"protocol_version": 2, "study_id": "mnist_scaleup_v1", "run_directory": "runs/mnist_scaleup_v1",
              "epochs": 20, "seed_offsets": [0, 1, 2, 3, 4], "variants": ["cnn", "fixed", "motion"],
              "train_sequences": 4000, "validation_sequences": 600, "test_sequences": 600,
              "test_block": 4, "pattern_counts": [128, 256], "bootstrap_repeats": 5000,
              "minimum_psnr_gain_db": 0.5, "split_manifest": "configs/mnist_scaleup_splits.json",
              "split_manifest_sha256": sha256_file(manifest_path), "cache_training_features": True}
    write_json(config_path, config)
    from experiment_protocol import protocol_report
    print(json.dumps(protocol_report(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()

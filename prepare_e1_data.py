"""Reserve E1 identities: fresh validation/test blocks, never evaluated before, plus a development block.

* train        4,000 official-train identities (same identities as scaleup_v1, new trajectories)
* development    600 official-train identities never used for training; for local engineering only
* validation     600 official-test identities, unused by any earlier experiment
* test           600 official-test identities, unused by any earlier experiment; evaluated once

Excluded from validation/test: the 1,352 identities already excluded by scaleup_v1, the 600 scaleup_v1
validation and 600 scaleup_v1 test identities, and anything a fresh scan of saved results finds.
No images are generated and no metric is computed here.
"""
from __future__ import annotations

import json

from gi_experiment import verify_assets, write_json
from prepare_scaleup_data import historical_exposure
from project_paths import PROJECT_ROOT, sha256_file
from train_graph_protocol_v2 import SEED, split_indices

MANIFEST = PROJECT_ROOT / "configs" / "e1_splits.json"
SCALEUP = PROJECT_ROOT / "configs" / "mnist_scaleup_splits.json"


def prepare():
    if MANIFEST.exists():
        print(f"{MANIFEST.name} already exists (sha256 {sha256_file(MANIFEST)}); not overwriting")
        return
    verify_assets()
    scaleup = json.loads(SCALEUP.read_text(encoding="utf-8"))
    excluded = set(scaleup["historically_exposed_test_indices"])
    scaleup_used = set(scaleup["splits"]["validation"]["mnist_indices"]) | set(scaleup["splits"]["test"]["mnist_indices"])
    scanned, audit = historical_exposure()
    excluded |= scaleup_used | scanned
    train_perm, test_perm = split_indices()
    available = [int(i) for i in test_perm if int(i) not in excluded]
    if len(available) < 1200:
        raise RuntimeError("Not enough unused official-test identities")
    train_ids = [int(i) for i in train_perm[:4000]]
    if train_ids != scaleup["splits"]["train"]["mnist_indices"]:
        raise RuntimeError("Training identities differ from scaleup_v1")
    splits = {
        "train": {"source": "official_train", "mnist_indices": train_ids, "trajectory_seed": SEED + 30701},
        "development": {"source": "official_train", "mnist_indices": [int(i) for i in train_perm[4000:4600]],
                        "trajectory_seed": SEED + 30709},
        "validation": {"source": "official_test", "mnist_indices": available[:600], "trajectory_seed": SEED + 30702},
        "test": {"source": "official_test", "mnist_indices": available[600:1200], "trajectory_seed": SEED + 30707},
    }
    manifest = {
        "schema_version": 1, "study_id": "e1_motion_pooling_v1", "data_seed": SEED,
        "selection": "Fixed permutation of official-test identities; drop every identity excluded by scaleup_v1, "
                     "the scaleup_v1 validation and test identities and any identity found in saved results; "
                     "first 600 remaining -> validation, next 600 -> final test",
        "historically_exposed_test_indices": sorted(excluded),
        "exclusion_counts": {"scaleup_v1_excluded": len(scaleup["historically_exposed_test_indices"]),
                             "scaleup_v1_validation_and_test": len(scaleup_used),
                             "total_unique": len(excluded)},
        "historical_exposure_sources": audit + [{"source": "configs/mnist_scaleup_splits.json (validation, test)",
                                                 "sha256": sha256_file(SCALEUP), "unique_identities": len(scaleup_used)}],
        "audit_scope": "Saved result records, reconstructed legacy samplers and the scaleup_v1 manifest; "
                       "unrecorded experiments cannot be audited",
        "splits": splits, "resolution": [64, 64], "frames": 8,
        "motion": "One digit, balanced speed strata 1-2/2-4/4-6 px/frame, reflection at +/-10 px",
        "data_interpolation": "bicubic",
        "raw_data": {item["path"]: item["sha256"] for item in
                     json.loads((PROJECT_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))["required_files"]
                     if item["path"].startswith("data/")},
        "final_test_images_generated": False, "final_test_metrics_evaluated": False,
    }
    for a in ("validation", "test"):
        assert not set(splits[a]["mnist_indices"]) & excluded
    assert not set(splits["validation"]["mnist_indices"]) & set(splits["test"]["mnist_indices"])
    assert not set(splits["train"]["mnist_indices"]) & set(splits["development"]["mnist_indices"])
    write_json(MANIFEST, manifest)
    print(json.dumps({"manifest": str(MANIFEST.relative_to(PROJECT_ROOT)), "sha256": sha256_file(MANIFEST),
                      **manifest["exclusion_counts"], "available_unused_test_identities": len(available)}, indent=2))


if __name__ == "__main__":
    prepare()

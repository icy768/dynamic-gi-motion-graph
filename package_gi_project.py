"""Build one folder archive containing code, data and weights, or archive results."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile

from gi_experiment import load_config, verify_assets, write_json
from project_paths import PROJECT_ROOT, code_version, project_path, sha256_file


def runtime_package():
    verify_assets()
    tracked = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8").split("\0")
    paths = {project_path(name) for name in tracked if name}
    manifest = json.loads((PROJECT_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
    paths.update(project_path(item["path"]) for item in manifest["required_files"])
    # Existing local weights for the archived experiments are small enough to keep together.
    for path in PROJECT_ROOT.rglob("*.pt"):
        relative = path.relative_to(PROJECT_ROOT)
        if relative.parts[0] not in {"runs", "tmp", ".local-backup", ".local-tools", "dist"}:
            paths.add(path)
    paths = sorted(path for path in paths if path.is_file())
    output = PROJECT_ROOT / "dist" / "gi_project_server.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    version = code_version()
    records = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative.startswith((".git/", "runs/", "dist/", "tmp/", ".local-tools/", ".local-backup/")):
                raise RuntimeError(f"Unexpected package member: {relative}")
            if path.suffix.lower() in {".pdf", ".pem", ".key"} or path.name.startswith(".env"):
                raise RuntimeError(f"Unexpected package member: {relative}")
            archive.write(path, "gi_project/" + relative)
            records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        archive.writestr("gi_project/CODE_VERSION.json", json.dumps(version, ensure_ascii=False, indent=2) + "\n")
        archive.writestr("gi_project/PACKAGE_MANIFEST.json", json.dumps({"code_version": version, "files": records}, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC error: {bad}")
    report = {"archive": output.relative_to(PROJECT_ROOT).as_posix(), "bytes": output.stat().st_size,
              "uncompressed_bytes": sum(item["bytes"] for item in records), "files": len(records) + 2,
              "sha256": sha256_file(output), "code_version": version, "top_level_directory": "gi_project"}
    write_json(PROJECT_ROOT / "dist" / "package_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return output


def result_package(config_path):
    config = load_config(config_path)
    run = project_path(config["run_directory"])
    if not (run / "final" / "results.json").is_file() or not (run / "efficiency.json").is_file():
        raise RuntimeError("Final evaluation and efficiency results must exist before packaging")
    output = PROJECT_ROOT / "dist" / "gi_results_block3.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                archive.write(path, "gi_results/" + path.relative_to(run).as_posix())
        archive.write(project_path(config_path), "gi_results/requested_config.json")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() or len(archive.namelist()) != len(set(archive.namelist())):
            raise RuntimeError("Result archive failed integrity or unique-member checks")
    print(f"results packaged: {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--config", default="configs/mnist_block3.json")
    args = parser.parse_args()
    if args.results:
        result_package(args.config)
    else:
        runtime_package()


if __name__ == "__main__":
    main()

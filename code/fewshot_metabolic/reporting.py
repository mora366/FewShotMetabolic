from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.name
    return value


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        json_value(payload), indent=2, sort_keys=True, ensure_ascii=False
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: str | Path) -> list[dict[str, object]]:
    base = Path(root)
    entries = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def duplicate_file_hashes(manifest: list[dict[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in manifest:
        digest = str(entry["sha256"])
        grouped.setdefault(digest, []).append(str(entry["path"]))
    return {digest: paths for digest, paths in grouped.items() if len(paths) > 1}


def forbidden_text_hits(
    root: str | Path, phrases: tuple[str, ...]
) -> dict[str, list[str]]:
    base = Path(root)
    results: dict[str, list[str]] = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pt", ".png", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        matches = [phrase for phrase in phrases if phrase.lower() in text]
        if matches:
            results[path.relative_to(base).as_posix()] = matches
    return results


def absolute_path_hits(root: str | Path) -> dict[str, list[int]]:
    base = Path(root)
    results = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pt"}:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        unix_marker = chr(47) + "Users" + chr(47)
        windows_marker = "C:" + chr(92) + "Users" + chr(92)
        matches = [
            index
            for index, line in enumerate(lines, start=1)
            if unix_marker in line or windows_marker in line
        ]
        if matches:
            results[path.relative_to(base).as_posix()] = matches
    return results


def integrity_payload(
    root: str | Path, verification_status: str, test_status: str
) -> dict[str, Any]:
    manifest = source_manifest(root)
    return {
        "verification_status": verification_status,
        "test_status": test_status,
        "files": manifest,
        "duplicate_files": duplicate_file_hashes(manifest),
        "absolute_path_hits": absolute_path_hits(root),
    }

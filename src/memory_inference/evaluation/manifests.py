from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunEnvironment:
    python_version: str
    python_executable: str
    platform: str
    hostname: str
    torch_version: str | None = None
    transformers_version: str | None = None


@dataclass(slots=True)
class RunManifest:
    benchmark: str
    reasoner: str
    policy_names: list[str]
    metrics: list[dict[str, Any]]
    config: dict[str, Any]
    created_at_utc: str
    git_commit: str | None = None
    git_dirty: bool = False
    environment: RunEnvironment | None = None


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2))


def build_manifest(
    *,
    benchmark: str,
    reasoner: str,
    policy_names: list[str],
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
    include_environment: bool = True,
) -> RunManifest:
    return RunManifest(
        benchmark=benchmark,
        reasoner=reasoner,
        policy_names=policy_names,
        metrics=metrics,
        config=config,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit() if include_environment else None,
        git_dirty=_git_dirty() if include_environment else False,
        environment=_environment() if include_environment else None,
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _environment() -> RunEnvironment:
    return RunEnvironment(
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        platform=platform.platform(),
        hostname=socket.gethostname(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


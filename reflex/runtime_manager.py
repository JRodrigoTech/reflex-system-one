"""Portable Python and backend-environment lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path
from typing import Callable

from reflex.operations import STAGING_MARGIN_BYTES, OperationLock, ensure_within, free_bytes, server_is_running
from reflex.lock_hash import lock_sha256 as canonical_lock_sha256
from reflex.paths import repository_root, runtime_root

_ENVIRONMENT_FAMILIES = ("core", "laya", "decider", "decision-cuda")


class RuntimeManagerError(RuntimeError):
    pass


def _lock_has_packages(lock_file: Path) -> bool:
    return any(
        re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\s*(?:==|~=|!=|<=|>=|<|>)", line.strip())
        for line in lock_file.read_text(encoding="utf-8").splitlines()
    )


def validate_runtime_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("python"), dict):
        raise RuntimeManagerError("Runtime manifest schema is invalid.")
    if type(value.get("staging_margin_bytes")) is not int or value["staging_margin_bytes"] < 1:
        raise RuntimeManagerError("Runtime manifest staging margin is invalid.")
    if isinstance(value.get("staging_margin_bytes"), bool):
        raise RuntimeManagerError("Runtime manifest staging margin is invalid.")
    environments = value.get("environments")
    if not isinstance(environments, dict) or set(environments) != set(_ENVIRONMENT_FAMILIES):
        raise RuntimeManagerError("Runtime environment inventory is invalid.")
    for family in _ENVIRONMENT_FAMILIES:
        environment = environments[family]
        if not isinstance(environment, dict):
            raise RuntimeManagerError("Runtime environment inventory is invalid.")
        if environment.get("lock_file") != f"requirements/{family}.lock":
            raise RuntimeManagerError("Runtime environment lock path is invalid.")
        digest = environment.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise RuntimeManagerError("Runtime environment lock hash is invalid.")
        if type(environment.get("estimated_unpacked_bytes")) is not int or environment["estimated_unpacked_bytes"] < 1:
            raise RuntimeManagerError("Runtime environment disk estimate is invalid.")
    spec = value["python"]
    if not all(isinstance(spec.get(key), str) and spec.get(key) for key in ("version", "release", "platform", "archive", "sha256")):
        raise RuntimeManagerError("Runtime manifest is missing a pinned Python field.")
    if type(spec.get("estimated_unpacked_bytes")) is not int or spec["estimated_unpacked_bytes"] < 1:
        raise RuntimeManagerError("Runtime manifest Python disk estimate is invalid.")
    if not spec["archive"].startswith("https://") or len(spec["sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in spec["sha256"]):
        raise RuntimeManagerError("Runtime manifest URL or SHA-256 is invalid.")
    return spec


def _read_runtime_manifest(root: Path) -> dict:
    try:
        manifest = json.loads((root / "manifests" / "runtime.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise RuntimeManagerError("The pinned runtime manifest is missing or invalid.") from None
    validate_runtime_manifest(manifest)
    return manifest


def _archive_unpacked_bytes(archive: Path) -> int:
    try:
        with tarfile.open(archive, "r:gz") as tar:
            total = 0
            for member in tar.getmembers():
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeManagerError("Runtime archive contains an unsupported link or device.")
                if member.isfile():
                    if member.size < 0:
                        raise RuntimeManagerError("Runtime archive contains an invalid file size.")
                    total += member.size
            return total
    except RuntimeManagerError:
        raise
    except (OSError, tarfile.TarError):
        raise RuntimeManagerError("Runtime archive could not be inspected before staging.") from None


def safe_extract_tar_gz(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != root and root not in candidate.parents:
                raise RuntimeManagerError("Runtime archive contains an unsafe path.")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeManagerError("Runtime archive contains an unsupported link or device.")
        tar.extractall(destination, filter="data")


def promote_directory(staged: Path, target: Path) -> None:
    staged = staged.resolve()
    target_parent = target.parent.resolve()
    if staged.parent != target_parent:
        raise RuntimeManagerError("Staged directory must be adjacent to the promoted runtime.")
    backup = target.with_name(target.name + ".old")
    if backup.exists() and target.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except OSError:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise RuntimeManagerError("Runtime promotion failed; the previous runtime was restored.") from None
    if backup.exists():
        shutil.rmtree(backup)


class PortableRuntimeManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or repository_root()).resolve()
        self.runtime = runtime_root(self.root)
        self.lock_path = self.runtime / "state" / "operation.lock"

    @staticmethod
    def _runtime_is_healthy(path: Path) -> bool:
        if not path.is_dir():
            return False
        for python_exe in path.rglob("python.exe"):
            try:
                version = subprocess.run([str(python_exe), "--version"], capture_output=True, text=True, timeout=10)
                pip = subprocess.run([str(python_exe), "-m", "pip", "--version"], capture_output=True, text=True, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if version.returncode == 0 and "Python 3." in version.stdout + version.stderr and pip.returncode == 0:
                return True
        return False

    def _recover_interrupted_swap(self, target: Path) -> None:
        backup = target.with_name(target.name + ".old")
        if not backup.exists():
            return
        current_valid = self._runtime_is_healthy(target)
        backup_valid = self._runtime_is_healthy(backup)
        if current_valid:
            shutil.rmtree(ensure_within(backup, self.runtime))
            return
        if not backup_valid:
            raise RuntimeManagerError("An interrupted runtime replacement has no verified recovery copy; existing data was preserved.")
        if target.exists():
            invalid_target = ensure_within(target, self.runtime)
            if invalid_target.is_dir():
                shutil.rmtree(invalid_target)
            else:
                invalid_target.unlink()
        os.replace(ensure_within(backup, self.runtime), ensure_within(target, self.runtime))

    def install_archive(self, archive: Path, expected_sha256: str, version: str, *, min_free_bytes: int | None = None) -> Path:
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeManagerError("Portable Python checksum does not match; nothing was promoted.")
        if server_is_running(self.root):
            raise RuntimeManagerError("Stop the Reflex server before replacing portable Python.")
        margin = _read_runtime_manifest(self.root)["staging_margin_bytes"] if min_free_bytes is None else min_free_bytes
        if type(margin) is not int or margin < 1:
            raise RuntimeManagerError("The portable Python staging margin is invalid.")
        target = self.runtime / "python"
        stage = self.runtime / "python.new"
        self.runtime.mkdir(parents=True, exist_ok=True)
        with OperationLock(self.lock_path, "install:portable-python"):
            self._recover_interrupted_swap(target)
            available_bytes = free_bytes(self.runtime)
            if available_bytes < margin:
                raise RuntimeManagerError("There is not enough free disk space for a safe runtime replacement.")
            extracted_bytes = _archive_unpacked_bytes(archive)
            if available_bytes < extracted_bytes + margin:
                raise RuntimeManagerError("There is not enough free disk space for the extracted runtime and safety margin.")
            if stage.exists():
                shutil.rmtree(ensure_within(stage, self.runtime))
            stage.mkdir(parents=True, exist_ok=True)
            try:
                safe_extract_tar_gz(archive, stage)
                candidates = list(stage.rglob("python.exe"))
                if not candidates:
                    raise RuntimeManagerError("Extracted runtime has no python.exe.")
                python_exe = candidates[0]
                completed = subprocess.run([str(python_exe), "--version"], capture_output=True, text=True, timeout=10)
                if completed.returncode or version not in completed.stdout + completed.stderr:
                    raise RuntimeManagerError("Extracted runtime version does not match the manifest.")
                subprocess.run([str(python_exe), "-m", "pip", "--version"], check=True, capture_output=True, timeout=20)
                promote_directory(stage, target)
            except Exception:
                if stage.exists():
                    shutil.rmtree(ensure_within(stage, self.runtime), ignore_errors=True)
                raise
        return target


class EnvironmentManager:
    """Rebuild one isolated backend family without touching other data."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or repository_root()).resolve()
        self.runtime = runtime_root(self.root)
        self.lock_path = self.runtime / "state" / "operation.lock"

    def install(self, family: str, python_exe: Path, lock_file: Path, lock_sha256: str) -> Path:
        if family not in {"core", "laya", "decider", "decision-cuda"}:
            raise RuntimeManagerError("Environment family is not in the Reflex registry.")
        if server_is_running(self.root):
            raise RuntimeManagerError("Stop Reflex before rebuilding an environment.")
        lock_file = lock_file.resolve()
        if not lock_file.is_file() or not lock_sha256:
            raise RuntimeManagerError("A populated, pinned family lock is required.")
        actual_hash = canonical_lock_sha256(lock_file)
        if actual_hash != lock_sha256.lower():
            raise RuntimeManagerError("Dependency lock hash does not match.")
        if not _lock_has_packages(lock_file):
            raise RuntimeManagerError("Dependency lock is empty; this backend has not passed qualification.")
        manifest = _read_runtime_manifest(self.root)
        family_manifest = manifest["environments"][family]
        expected_lock = (self.root / family_manifest["lock_file"]).resolve()
        if lock_file != expected_lock or family_manifest["sha256"] != actual_hash:
            raise RuntimeManagerError("Dependency lock does not match the pinned environment manifest.")
        required_free_bytes = 2 * family_manifest["estimated_unpacked_bytes"] + manifest["staging_margin_bytes"]
        envs = self.runtime / "envs"
        pip_cache = ensure_within(self.runtime / "cache" / "pip", self.runtime)
        pip_environment = os.environ.copy()
        pip_environment["PIP_CACHE_DIR"] = str(pip_cache)
        target = envs / family
        stage = envs / f"{family}.new"
        with OperationLock(self.lock_path, f"install:environment:{family}"):
            if free_bytes(self.runtime) < required_free_bytes:
                raise RuntimeManagerError("There is not enough free disk space for the staged environment, package cache, and safety margin.")
            envs.mkdir(parents=True, exist_ok=True)
            pip_cache.mkdir(parents=True, exist_ok=True)
            if stage.exists():
                shutil.rmtree(ensure_within(stage, envs))
            try:
                subprocess.run([str(python_exe), "-m", "venv", str(stage)], check=True, timeout=180)
                env_python = stage / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                subprocess.run([
                    str(env_python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--require-hashes", "-r", str(lock_file),
                ], check=True, timeout=1800, env=pip_environment)
                subprocess.run([
                    str(env_python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-index", "--no-deps", "--no-build-isolation", "-e", str(self.root),
                ], check=True, timeout=180, env=pip_environment)
                subprocess.run([str(env_python), "-m", "pip", "check"], check=True, timeout=60, env=pip_environment)
                (stage / "reflex-lock.json").write_text(json.dumps({"sha256": actual_hash, "family": family}) + "\n", encoding="utf-8")
                promote_directory(stage, target)
            except Exception:
                if stage.exists():
                    shutil.rmtree(ensure_within(stage, envs), ignore_errors=True)
                raise
        return target


def install_family_environment(family: str, root: Path | None = None) -> Path:
    """Install one native Windows backend family from its checked-in hash lock."""
    root = (root or repository_root()).resolve()
    if family not in {"core", "laya", "decider", "decision-cuda"}:
        raise RuntimeManagerError("This backend environment must be installed by its platform-specific launcher.")
    portable_python = runtime_root(root) / "python" / "python" / "python.exe"
    if not portable_python.is_file():
        raise RuntimeManagerError("The Reflex-owned portable Python runtime is not installed. Run the launcher first.")
    lock = root / "requirements" / f"{family}.lock"
    if not lock.is_file() or not _lock_has_packages(lock):
        raise RuntimeManagerError(f"The {family} dependency lock is not populated and verified yet.")
    digest = canonical_lock_sha256(lock)
    return EnvironmentManager(root).install(family, portable_python, lock, digest)

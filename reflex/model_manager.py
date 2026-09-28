"""Pinned Hugging Face model lifecycle operations."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from reflex.operations import STAGING_MARGIN_BYTES, OperationLock, ensure_within, free_bytes, server_is_running
from reflex.paths import model_path, model_root, repository_root
from reflex.registry import ModelSpec, load_registry

MANIFEST_NAME = "reflex-model.json"
STAGING_MARGIN = STAGING_MARGIN_BYTES


class ModelManagerError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, root: Path | None = None, registry: dict[str, ModelSpec] | None = None) -> None:
        self.root = (root or repository_root()).resolve()
        self.registry = registry or load_registry(self.root)
        self.lock_path = self.root / "runtime" / "state" / "operation.lock"

    def destination(self, model_id: str, variant: str | None = None) -> Path:
        spec = self.registry.get(model_id)
        if spec is None:
            raise ModelManagerError("Model ID is not in the Reflex registry.")
        if spec.variants and variant not in spec.variants:
            raise ModelManagerError("Selected variant is not in the Reflex registry.")
        if not spec.variants and variant is not None:
            raise ModelManagerError("This model does not have variants.")
        return ensure_within(model_path(model_id, variant, self.root), model_root(self.root))

    def info(self, model_id: str, variant: str | None = None, *, verify_hashes: bool = False) -> dict[str, Any]:
        spec = self.registry.get(model_id)
        if spec is None:
            raise ModelManagerError("Model ID is not in the Reflex registry.")
        selected_variant = variant or spec.default_variant
        support_status = spec.support_status(selected_variant)
        from reflex.qualification import qualified_devices
        qualified_for = qualified_devices(self.root, spec, selected_variant if spec.model_id == "laya" else None)
        qualified = bool(qualified_for)
        path = self.destination(model_id, selected_variant)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.is_file():
            return {
                "model": model_id,
                "variant": selected_variant,
                "status": "NOT INSTALLED",
                "support": support_status,
                "qualified": qualified,
                "qualified_devices": list(qualified_for),
                "revision_pinned": spec.pinned,
            }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"model": model_id, "variant": selected_variant, "status": "BROKEN",
                    "support": support_status, "qualified": qualified,
                    "qualified_devices": list(qualified_for)}
        verified, reason = self._verify_manifest(
            path, manifest, spec, variant=selected_variant, verify_hashes=verify_hashes,
        )
        return {
            "model": model_id,
            "variant": selected_variant,
            "status": "READY" if verified else "BROKEN",
            "support": support_status,
            "qualified": qualified,
            "qualified_devices": list(qualified_for),
            "revision": manifest.get("revision") if isinstance(manifest, dict) else None,
            "reason": reason,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _verify_manifest(
        self,
        path: Path,
        manifest: object,
        spec: ModelSpec,
        *,
        variant: str | None = None,
        verify_hashes: bool = False,
    ) -> tuple[bool, str]:
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
            return False, "Installed manifest schema is invalid."
        if manifest.get("model_id") != spec.model_id or manifest.get("repository") != spec.repository:
            return False, "Installed manifest does not match the registry."
        if manifest.get("source_url") != spec.source_url or manifest.get("license") != spec.license:
            return False, "Installed source or license metadata does not match the registry."
        expected_variant = variant if spec.variants else None
        if manifest.get("backend") != spec.backend or manifest.get("variant") != expected_variant:
            return False, "Installed backend or variant does not match the selected checkpoint."
        if not spec.pinned or manifest.get("revision") != spec.revision:
            return False, "Installed revision does not match a pinned registry revision."
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            return False, "Installed file inventory is empty."
        for record in files:
            if not isinstance(record, dict):
                return False, "Installed file inventory is invalid."
            relative = record.get("path")
            if not isinstance(relative, str) or not relative:
                return False, "Installed file inventory contains an invalid path."
            try:
                candidate = ensure_within(path / relative, path)
            except ValueError:
                return False, "Installed manifest contains an unsafe path."
            if not candidate.is_file():
                return False, "An installed model file is missing."
            expected_size = record.get("size")
            if not isinstance(expected_size, int) or candidate.stat().st_size != expected_size:
                return False, "An installed model file has an unexpected size."
            expected_hash = record.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash.lower()):
                return False, "Installed file inventory has no valid checksum."
            if verify_hashes and self._sha256(candidate) != expected_hash:
                return False, "An installed model file failed SHA-256 verification."
        for required in spec.required_files:
            if not (path / required).is_file():
                return False, "A required model file is missing."
        return True, "All recorded model files are present."

    def verify(self, model_id: str, variant: str | None = None) -> bool:
        info = self.info(model_id, variant, verify_hashes=True)
        return info.get("status") == "READY"

    def download(self, model_id: str, variant: str | None = None, *, timeout_seconds: int = 1800) -> Path:
        spec = self.registry.get(model_id)
        if spec is None:
            raise ModelManagerError("Model ID is not in the Reflex registry.")
        if not spec.pinned:
            raise ModelManagerError("Download blocked: this model has no exact 40-character commit revision in the trusted registry.")
        if not spec.required_files:
            raise ModelManagerError("Download blocked: the registry has no verified required-file contract for this checkpoint.")
        if server_is_running(self.root):
            raise ModelManagerError("Stop the Reflex server before changing model files.")
        destination = self.destination(model_id, variant or spec.default_variant)
        root = model_root(self.root)
        root.mkdir(parents=True, exist_ok=True)
        estimate_key = variant or "default"
        expected_bytes = spec.estimated_bytes.get(estimate_key)
        if not expected_bytes:
            raise ModelManagerError("The registry has no verified disk estimate for this checkpoint.")
        with OperationLock(self.lock_path, f"download:{model_id}"):
            selected_variant = variant or spec.default_variant
            self._recover_interrupted_swap(spec, destination, selected_variant)
            stage_parent = destination.with_name(destination.name + ".partial")
            if stage_parent.exists():
                shutil.rmtree(ensure_within(stage_parent, root))
            old_bytes = sum(file.stat().st_size for file in destination.rglob("*") if file.is_file()) if destination.exists() else 0
            required_free = 2 * expected_bytes + old_bytes + STAGING_MARGIN
            if free_bytes(root) < required_free:
                raise ModelManagerError("There is not enough free disk space for the new staging copy, recovery copy, and safety margin.")
            try:
                return self._download_locked(spec, destination, selected_variant, timeout_seconds)
            except Exception:
                if stage_parent.exists():
                    shutil.rmtree(ensure_within(stage_parent, root), ignore_errors=True)
                raise

    def _directory_is_verified(self, path: Path, spec: ModelSpec, variant: str | None) -> bool:
        try:
            manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
            return self._verify_manifest(path, manifest, spec, variant=variant, verify_hashes=True)[0]
        except (OSError, ValueError, TypeError):
            return False

    def _recover_interrupted_swap(self, spec: ModelSpec, destination: Path, variant: str | None) -> None:
        """Restore a verified .old checkpoint before discarding stale staging data."""
        backup = destination.with_name(destination.name + ".old")
        if not backup.exists():
            return
        if self._directory_is_verified(destination, spec, variant):
            shutil.rmtree(ensure_within(backup, model_root(self.root)))
            return
        if not self._directory_is_verified(backup, spec, variant):
            raise ModelManagerError("An interrupted model replacement has no verified recovery copy; existing data was preserved.")
        if destination.exists():
            invalid_destination = ensure_within(destination, model_root(self.root))
            if invalid_destination.is_dir():
                shutil.rmtree(invalid_destination)
            else:
                invalid_destination.unlink()
        os.replace(ensure_within(backup, model_root(self.root)), ensure_within(destination, model_root(self.root)))

    def _download_locked(self, spec: ModelSpec, destination: Path, variant: str | None, timeout_seconds: int) -> Path:
        try:
            cache = self.root / "runtime" / "cache" / "huggingface"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache)
            os.environ["HF_HUB_CACHE"] = str(cache / "hub")
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(max(1, min(timeout_seconds, 300)))
            os.environ["HF_HUB_ETAG_TIMEOUT"] = str(max(1, min(timeout_seconds, 30)))
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ModelManagerError("Hugging Face Hub is missing from the selected installer environment.")
        stage_parent = destination.with_name(destination.name + ".partial")
        source_subfolder = spec.variant_subfolders.get(variant, "") if variant else ""
        staged_model = stage_parent / source_subfolder if source_subfolder else stage_parent
        if stage_parent.exists():
            shutil.rmtree(ensure_within(stage_parent, model_root(self.root)))
        stage_parent.mkdir(parents=True, exist_ok=True)
        patterns = list(spec.required_files)
        if spec.model_id == "laya":
            patterns.append("README.md")
        elif spec.backend == "decider":
            patterns.extend(["README.md", "decider/*.py"])
        elif spec.backend == "lux":
            patterns.extend(["code/*.py", "src/**/*.py"])
        elif spec.backend == "semantic-router":
            patterns.extend(["code/*.py"])
        if source_subfolder:
            allow_patterns = [f"{source_subfolder}/{pattern}" for pattern in patterns]
            allow_patterns.extend(["README.md", "LICENSE"])
        else:
            allow_patterns = patterns + ["README.md", "LICENSE"]
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        for attempt in range(3):
            if time.monotonic() >= deadline:
                break
            try:
                snapshot_download(
                    repo_id=spec.repository,
                    revision=spec.revision,
                    local_dir=str(stage_parent),
                    allow_patterns=allow_patterns,
                    etag_timeout=min(30, timeout_seconds),
                    max_workers=4,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
        if last_error is not None:
            raise ModelManagerError("Model download failed; the existing verified model was preserved.") from None
        if source_subfolder:
            try:
                staged_model = ensure_within(staged_model, stage_parent)
            except ValueError:
                raise ModelManagerError("Downloaded snapshot contained an unsafe checkpoint variant path.") from None
            if not staged_model.is_dir():
                raise ModelManagerError("Downloaded snapshot did not contain the selected checkpoint variant.")
        target = staged_model
        if not target.is_dir():
            target = stage_parent
        try:
            target = ensure_within(target, stage_parent)
        except ValueError:
            raise ModelManagerError("Downloaded snapshot contained a path outside its staging directory.") from None
        if source_subfolder:
            for upstream_name in ("README.md", "LICENSE"):
                try:
                    upstream_file = ensure_within(stage_parent / upstream_name, stage_parent)
                    metadata_target = ensure_within(target / ("UPSTREAM-" + upstream_name), target)
                except ValueError:
                    raise ModelManagerError("Downloaded upstream metadata contains an unsafe path.") from None
                if upstream_file.is_file():
                    shutil.copy2(upstream_file, metadata_target)

        relative_files = []
        try:
            for path in target.rglob("*"):
                if path.name == MANIFEST_NAME or ".cache" in path.parts:
                    continue
                safe_path = ensure_within(path, target)
                if safe_path.is_file():
                    relative_files.append(path.relative_to(target).as_posix())
        except ValueError:
            raise ModelManagerError("Downloaded model contains a path outside its staging directory.") from None
        relative_files.sort()

        try:
            required_paths = [ensure_within(target / required, target) for required in spec.required_files]
        except ValueError:
            raise ModelManagerError("The registry required-file contract contains an unsafe path.") from None
        if not relative_files or any(not required.is_file() for required in required_paths):
            raise ModelManagerError("Downloaded model did not pass the required-file check.")
        files = []
        for relative in relative_files:
            try:
                file_path = ensure_within(target / relative, target)
            except ValueError:
                raise ModelManagerError("Downloaded model contains a path outside its staging directory.") from None
            files.append({"path": relative, "size": file_path.stat().st_size, "sha256": self._sha256(file_path)})
        manifest = {
            "schema_version": 1,
            "model_id": spec.model_id,
            "repository": spec.repository,
            "source_url": spec.source_url,
            "license": spec.license,
            "revision": spec.revision,
            "backend": spec.backend,
            "variant": variant,
            "files": files,
            "installed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (target / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        valid, _ = self._verify_manifest(target, manifest, spec, variant=variant, verify_hashes=True)
        if not valid:
            raise ModelManagerError("Staged model failed verification; the existing model was preserved.")
        backup = destination.with_name(destination.name + ".old")
        if backup.exists():
            shutil.rmtree(ensure_within(backup, model_root(self.root)))
        if destination.exists():
            os.replace(destination, backup)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, destination)
        except OSError:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise ModelManagerError("Model promotion failed; the previous model was restored.") from None
        if backup.exists():
            shutil.rmtree(backup)
        if stage_parent.exists():
            shutil.rmtree(stage_parent)
        return destination

    def delete(self, model_id: str, variant: str | None = None) -> None:
        if server_is_running(self.root):
            raise ModelManagerError("Stop the Reflex server before deleting model files.")
        destination = self.destination(model_id, variant or self.registry[model_id].default_variant)
        if destination.exists():
            with OperationLock(self.lock_path, f"delete:{model_id}"):
                shutil.rmtree(ensure_within(destination, model_root(self.root)))

    def repair(self, model_id: str, variant: str | None = None) -> Path:
        # A verified model is already repaired; broken installs are safely replaced from a pinned source.
        if self.verify(model_id, variant):
            return self.destination(model_id, variant or self.registry[model_id].default_variant)
        return self.download(model_id, variant)

    def select_active(self, model_id: str, config: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
        from reflex.config import save_config

        spec = self.registry.get(model_id)
        if spec is None:
            raise ModelManagerError("Model ID is not in the Reflex registry.")
        selected_variant = variant or spec.default_variant
        if spec.support_status(selected_variant if model_id == "laya" else None) != "SUPPORTED":
            raise ModelManagerError("Only qualified supported models can be selected for normal inference.")
        if server_is_running(self.root):
            raise ModelManagerError("Stop the Reflex server before changing the active model.")
        from reflex.qualification import qualification_record
        selected_device = config.get("device_policy", "auto")
        if selected_device == "auto":
            has_qualification = any(
                qualification_record(
                    self.root, spec, selected_variant if model_id == "laya" else None, candidate
                ) is not None
                for candidate in spec.devices
            )
        else:
            has_qualification = qualification_record(
                self.root, spec, selected_variant if model_id == "laya" else None, selected_device
            ) is not None
        if not has_qualification:
            raise ModelManagerError("The selected checkpoint has no qualification record matching its pinned model and environment.")
        if not self.verify(model_id, selected_variant):
            raise ModelManagerError("The selected model is not installed and verified.")
        with OperationLock(self.lock_path, f"select:{model_id}"):
            updated = {**config, "active_model": model_id}
            if model_id == "laya":
                updated["laya_variant"] = selected_variant
            save_config(updated, self.root)
        return updated

"""Structured health checks and conservative repair previews."""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from reflex.config import DEFAULT_CONFIG, load_config, save_config
from reflex.errors import ReflexError
from reflex.model_manager import ModelManager, STAGING_MARGIN
from reflex.operations import server_is_running
from reflex.paths import config_path, repository_root, runtime_root
from reflex.registry import load_registry

_EXTENDED_SMOKE_CODE = r'''
import json
import math
import socket
import sys
import threading
import time
from urllib.request import Request, urlopen

import uvicorn
from reflex.api import create_app
from reflex.backends.factory import load_service

service = None
server = None
server_thread = None
listener = None
server_started = False
try:
    service = load_service()
    service.prepare()
    request = {
        "state": "The customer was charged twice for the same invoice.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this issue?",
                "criteria": {
                    "billing": "Payments, invoices, and duplicate charges.",
                    "account": "Login and profile access.",
                },
            },
        },
    }
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    app = create_app(service, host="127.0.0.1", admission_capacity=2, max_body_bytes=65536,
                     shutdown_grace_seconds=5.0)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, workers=1, log_level="critical",
        access_log=False, timeout_graceful_shutdown=5.0,
    ))
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 60.0
    while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("temporary server did not become ready")
    server_started = True
    base = "http://127.0.0.1:" + str(port)
    with urlopen(base + "/health", timeout=5) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health.get("ready") is not True or health.get("model") != service.active_model:
        raise RuntimeError("temporary server health mismatch")
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    http_request = Request(base + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urlopen(http_request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    answer = result["answers"]["route"]
    probabilities = list(answer["probabilities"].values())
    if result["model"] != service.active_model or answer["choice"] != "billing":
        raise RuntimeError("golden mismatch")
    if (not probabilities or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)):
        raise RuntimeError("invalid probability output")
    print(json.dumps({"status": "READY", "model": result["model"], "server_smoke": True}, separators=(",", ":")))
except Exception:
    print(json.dumps({"status": "BROKEN"}, separators=(",", ":")))
    sys.exit(1)
finally:
    if server is not None:
        server.should_exit = True
    if server_thread is not None:
        server_thread.join(timeout=10.0)
    if listener is not None:
        listener.close()
    if service is not None and not (server_started and server_thread is not None and not server_thread.is_alive()):
        service.backend.close()
if server_thread is not None and server_thread.is_alive():
    print(json.dumps({"status": "BROKEN"}, separators=(",", ":")))
    sys.exit(1)
'''

_TORCH_DEVICE_PROBE = (
    "import json,torch; available=bool(torch.cuda.is_available()); "
    "print(json.dumps({'cuda_available': available, "
    "'bf16_supported': bool(torch.cuda.is_bf16_supported()) if available else False}))"
)
_OPTIONAL_KERNEL_PROBE = r'''
import importlib.metadata as metadata
import json

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def probe(statement):
    try:
        exec(statement, {})
        return {"ok": True, "reason": None}
    except ModuleNotFoundError as exc:
        return {"ok": False, "reason": "missing module: " + str(exc.name)}
    except Exception as exc:
        return {"ok": False, "reason": "import failed: " + type(exc).__name__}

result = {
    "transformers": version("transformers"),
    "flash_linear_attention": version("flash-linear-attention"),
    "fla_core": version("fla-core"),
    "causal_conv1d": version("causal-conv1d"),
    "fla_operator": probe("from fla.ops.gated_delta_rule import chunk_gated_delta_rule"),
    "causal_conv1d_operator": probe("from causal_conv1d import causal_conv1d_fn"),
}
print(json.dumps(result))
'''
MODEL_ENVIRONMENT_PROBE_TIMEOUT_SECONDS = 30


def _optional_cuda_kernel_check(python: Path, root: Path) -> tuple[str, str]:
    """Report optional Transformers kernel imports without treating fallback as fatal."""
    try:
        checked = subprocess.run(
            [str(python), "-c", _OPTIONAL_KERNEL_PROBE], cwd=root,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=MODEL_ENVIRONMENT_PROBE_TIMEOUT_SECONDS, check=False,
        )
        details = json.loads(checked.stdout.strip().splitlines()[-1]) if checked.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        details = None
    if not isinstance(details, dict):
        return "WARN", "Optional Transformers kernel imports could not be checked; server warmup will report any reference fallback before listening."
    imports = {name: details.get(name, {}) for name in ("fla_operator", "causal_conv1d_operator")}
    missing = [name for name, outcome in imports.items() if not isinstance(outcome, dict) or outcome.get("ok") is not True]
    versions = (f"Transformers {details.get('transformers') or 'not installed'}; "
                f"flash-linear-attention {details.get('flash_linear_attention') or 'not installed'}; "
                f"causal-conv1d {details.get('causal_conv1d') or 'not installed'}.")
    if missing:
        unavailable = []
        for name in missing:
            label = "FLA gated-delta operator" if name == "fla_operator" else "causal-conv1d operator"
            reason = imports[name].get("reason") if isinstance(imports[name], dict) else None
            unavailable.append(f"{label} ({reason or 'not importable'})")
        return "WARN", (versions + " If the selected model invokes these operations, Transformers uses its PyTorch reference fallback for "
                         + ", ".join(unavailable) + "; this is slower, not a fatal inference error. "
                         "The startup readiness request exercises lazy dispatch before the server listens.")
    return "READY", (versions + " Optional operators import successfully; actual dispatch is exercised by "
                     "startup preparation and is not inferred from package presence alone.")


def _extended_inference_check(root: Path, spec, variant: str | None, info: dict[str, Any], server_running: bool) -> tuple[str, str]:
    if spec.support_status(variant if spec.model_id == "laya" else None) != "SUPPORTED":
        return "EXPERIMENTAL", f"{spec.model_id} is not qualified for normal inference."
    if info.get("status") != "READY":
        status = info.get("status", "BROKEN")
        return str(status), f"{spec.model_id} model files must be READY before an extended load check."
    if info.get("qualified") is not True:
        return "EXPERIMENTAL", f"{spec.model_id} has no qualification matching its pinned model and runtime lock."
    if server_running:
        return "WARN", "Extended model loading was skipped because the Reflex server is already running."

    try:
        executable = runtime_root(root) / "envs" / spec.environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not executable.is_file():
            return "NOT INSTALLED", f"The isolated {spec.environment} environment is not installed."
        environment = dict(os.environ)
        environment.update({
            "REFLEX_ROOT": str(root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        checked = subprocess.run(
            [str(executable), "-c", _EXTENDED_SMOKE_CODE],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "BROKEN", "The active-model load or golden request exceeded the 30-minute extended-check limit."
    except (OSError, ValueError, RuntimeError):
        return "BROKEN", "The isolated backend could not complete its extended load check."

    result = None
    for line in reversed(checked.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and candidate.get("status") in {"READY", "BROKEN"}:
            result = candidate
            break
    if (checked.returncode or result is None or result.get("status") != "READY"
            or result.get("model") != spec.model_id or result.get("server_smoke") is not True):
        return "BROKEN", "The active model failed its isolated load, billing-route golden check, or temporary-server health check."
    return "READY", f"{spec.model_id} loaded in its isolated Windows backend runtime and passed the billing-route golden and temporary-loopback server smoke."


def inspect(root: Path | None = None, extended: bool = False) -> list[dict[str, Any]]:
    root = (root or repository_root()).resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, message: str, repair: str | None = None) -> None:
        checks.append({"name": name, "status": status, "message": message, "repair": repair})

    active_model = None
    active_variant = None
    active_info: dict[str, Any] = {}
    running = False
    try:
        usage = shutil.disk_usage(root)
        with tempfile.TemporaryFile(dir=root):
            pass
        add("filesystem", "READY", f"Repository storage is available ({usage.free // (1024**3)} GiB free).")
        if usage.free < STAGING_MARGIN:
            add("disk_space", "WARN", "Less than 2 GiB is free; model and runtime replacement may require more space for safe staging.")
        else:
            add("disk_space", "READY", "At least 2 GiB is free for safe runtime/model staging.")
    except OSError:
        add("filesystem", "BROKEN", "Repository storage cannot be inspected or written.", "Check the repository drive and write permissions.")

    portable = runtime_root(root) / "python" / "python" / "python.exe"
    add("portable_python", "READY" if portable.is_file() else "NOT INSTALLED",
        "Reflex portable Python is installed." if portable.is_file() else "Reflex portable Python is not installed.",
        None if portable.is_file() else "Run the Reflex launcher to install the pinned runtime.")

    try:
        config = load_config(root)
        registry = load_registry(root)
        spec = registry[config["active_model"]]
        active_model = spec
        manager = ModelManager(root, registry)
        active_variant = config["laya_variant"] if spec.model_id == "laya" else None
        info = manager.info(spec.model_id, active_variant, verify_hashes=extended)
        active_info = info
        qualified = info.get("qualified") is True
        add("active_model_files", info["status"], f"{spec.model_id}: {info['status'].lower()}; qualification status is {info.get('support', spec.status)}.",
            "Open Model Manager to verify or repair the selected model.")
        static_support = spec.support_status(active_variant)
        support_status = "READY" if qualified else ("WARN" if static_support == "SUPPORTED" else static_support)
        add("model_support", support_status,
            "The selected model and variant match a passing qualification record." if qualified else "This exact model variant has no passing release qualification record.")
        if spec.backend == "lux":
            add("gpu_runtime", "EXPERIMENTAL", "Lux runtime qualification has not passed on this machine.")
        running = server_is_running(root)
        add("server_process", "READY" if running else "STOPPED", "A Reflex server process is running." if running else "No Reflex server process is marked running.")
        if not running:
            try:
                with socket.socket(socket.AF_INET6 if ":" in config["host"] else socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind((config["host"], config["port"]))
                add("server_port", "READY", f"{config['host']}:{config['port']} is available.")
            except OSError:
                add("server_port", "BROKEN", f"{config['host']}:{config['port']} cannot be bound.", "Choose a free loopback port in Settings.")
        else:
            add("server_port", "WARN", "Port check skipped while the Reflex server is running.")
    except Exception:
        add("configuration", "BROKEN", "Configuration or the static model registry could not be read.", "Run Doctor repair for configuration.")

    env_root = runtime_root(root) / "envs"
    for family in ("core", "laya", "decider", "decision-cuda"):
        env_path = env_root / family
        python = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            add(f"environment_{family}", "BROKEN" if env_path.exists() else "NOT INSTALLED",
                f"{family} environment executable {'is missing' if env_path.exists() else 'is not installed'}.",
                f"Rebuild the isolated {family} environment from its verified lock.")
            continue
        if family == "core":
            imports = "fastapi,uvicorn,huggingface_hub,reflex"
        elif family == "laya":
            imports = "torch,laya"
        elif family == "decision-cuda":
            imports = "torch,transformers,safetensors,tokenizers,triton,fla,reflex"
        else:
            # Decider's reviewed API implementation ships inside each pinned
            # model snapshot. It is deliberately not imported from an
            # unrelated environment-wide `decider` package.
            imports = "torch,transformers,reflex"
        try:
            lock = root / "requirements" / f"{family}.lock"
            marker = env_path / "reflex-lock.json"
            verify = (
                "import hashlib,json,pathlib,sys; "
                "marker=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "lock=pathlib.Path(sys.argv[2]); "
                "raw=lock.read_bytes(); normalized=raw.replace(b'\\r\\n',b'\\n').replace(b'\\r',b'\\n'); "
                "crlf=normalized.replace(b'\\n',b'\\r\\n'); "
                "valid={hashlib.sha256(value).hexdigest() for value in (raw,normalized,crlf)}; "
                "assert marker.get('sha256') in valid; "
                + ("import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); " if family == "decision-cuda" else "")
                + f"import {imports}"
            )
            checked = subprocess.run(
                [str(python), "-c", verify, str(marker), str(lock)],
                capture_output=True,
                timeout=10 if family == "core" else MODEL_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
            )
            healthy = checked.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            healthy = False
        add(f"environment_{family}", "READY" if healthy else "BROKEN",
            f"{family} environment imports {'passed' if healthy else 'failed'}.",
            None if healthy else f"Rebuild the isolated {family} environment from its verified lock.")

    if active_model is None:
        add("gpu_torch", "UNAVAILABLE", "The active model runtime could not be resolved for a Torch device check.")
    else:
        family = active_model.environment
        python = env_root / family / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            add("gpu_torch", "UNAVAILABLE", f"The active {family} model environment is not installed.")
        else:
            try:
                probe = subprocess.run(
                    [str(python), "-c", _TORCH_DEVICE_PROBE],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=MODEL_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
                    check=False,
                )
                result = None
                for line in reversed(probe.stdout.splitlines()):
                    try:
                        candidate = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(candidate, dict) and isinstance(candidate.get("cuda_available"), bool):
                        result = candidate
                        break
                if probe.returncode or result is None:
                    add("gpu_torch", "BROKEN", f"Torch device status could not be read in the active {family} environment.",
                        f"Rebuild the isolated {family} environment from its verified lock.")
                else:
                    available = result["cuda_available"]
                    add("gpu_torch", "READY" if available else "WARN",
                        f"The active {family} environment reports CUDA available: {available}.")
            except (OSError, subprocess.TimeoutExpired):
                add("gpu_torch", "BROKEN", f"Torch device status could not be read in the active {family} environment.",
                    f"Rebuild the isolated {family} environment from its verified lock.")
    if active_model is not None:
        kernel_python = env_root / active_model.environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not kernel_python.is_file():
            add("optional_transformers_kernels", "UNAVAILABLE",
                f"The active {active_model.environment} environment is not installed; optional kernel imports cannot be checked.")
        else:
            status, message = _optional_cuda_kernel_check(kernel_python, root)
            add("optional_transformers_kernels", status, f"Active {active_model.environment} environment: {message}")
    if extended:
        if active_model is None:
            add("extended_inference", "BROKEN", "The active model could not be resolved for an extended check.")
        else:
            status, message = _extended_inference_check(root, active_model, active_variant, active_info, running)
            add("extended_inference", status, message)
    return checks


def repair_preview(checks: list[dict[str, Any]]) -> list[dict[str, str]]:
    proposals: list[dict[str, str]] = []
    for check in checks:
        if check["status"] in {"BROKEN", "NOT INSTALLED"} and check.get("repair"):
            component = check["name"]
            proposals.append({
                "action_id": f"repair.{component}",
                "component": component,
                "action": check["repair"],
            })
    return proposals


def repair_configuration(root: Path | None = None) -> dict[str, Any]:
    """Reset only an unreadable/invalid config, preserving a dated backup."""
    root = (root or repository_root()).resolve()
    path = config_path(root)
    try:
        return load_config(root)
    except ReflexError:
        if path.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = path.with_name(f"{path.name}.broken-{stamp}")
            os.replace(path, backup)
    save_config(DEFAULT_CONFIG, root)
    return dict(DEFAULT_CONFIG)

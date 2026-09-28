"""Small console wizard using injectable streams for deterministic tests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from reflex.branding import render_main_menu
from reflex.config import DEFAULT_CONFIG, load_config, save_config
from reflex.doctor import inspect, repair_configuration, repair_preview
from reflex.lock_hash import lock_hash_matches
from reflex.model_manager import ModelManager
from reflex.paths import repository_root
from reflex.reinstall import clean_reinstall, ReinstallError
from reflex.registry import load_registry
from reflex.runtime_manager import install_family_environment


def _device_label(spec) -> str:
    names = {"cpu": "CPU", "cuda": "CUDA"}
    return "/".join(names.get(device, device.upper()) for device in spec.devices)


def _read(source: TextIO, sink: TextIO, prompt: str) -> str:
    sink.write(prompt)
    sink.flush()
    value = source.readline()
    if value == "":
        raise EOFError
    return value.strip()


def _menu_index(value: str, item_count: int) -> int | None:
    try:
        selected = int(value)
    except ValueError:
        return None
    return selected - 1 if 1 <= selected <= item_count else None


def _family_runtime_status(root: Path, family: str) -> str:
    """Return READY, BROKEN, or NOT INSTALLED for one backend-family runtime."""
    env = root / "runtime" / "envs" / family
    marker = env / "reflex-lock.json"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    lock = root / "requirements" / f"{family}.lock"
    if not marker.is_file() or (executable is not None and not executable.is_file()):
        return "NOT INSTALLED"
    if not lock.is_file():
        return "BROKEN"
    try:
        installed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "BROKEN"
    actual = installed.get("sha256", installed.get("lock_sha256"))
    if not lock_hash_matches(actual, lock):
        return "BROKEN"
    package_versions: dict[str, str] = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)", line.strip())
        if match:
            package_versions[match.group(1)] = match.group(2)
    if not package_versions:
        return "BROKEN"

    imports = {
        "core": ["fastapi", "uvicorn", "huggingface_hub", "reflex"],
        "laya": ["torch", "transformers", "laya", "reflex"],
        "decider": ["torch", "transformers", "safetensors", "reflex"],
        "decision-cuda": ["torch", "transformers", "fla", "triton", "safetensors", "reflex"],
    }.get(family, ["reflex"])
    probe = (
        "import importlib,importlib.metadata as metadata,json,subprocess,sys; "
        "expected=json.loads(sys.argv[1]); imports=json.loads(sys.argv[2]); "
        "actual={name:metadata.version(name) for name in expected}; "
        "assert actual==expected, 'locked package versions differ'; "
        "[importlib.import_module(name) for name in imports]; "
        + ("import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); " if family == "decision-cuda" else "")
        + "check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=15); "
        + "sys.exit(check.returncode)"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", probe, json.dumps(package_versions), json.dumps(imports)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "BROKEN"
    return "READY" if result.returncode == 0 else "BROKEN"


def _ensure_family_environment(spec, sink: TextIO, root: Path) -> bool:
    """Make the selected backend-family runtime usable without touching model files."""
    status = _family_runtime_status(root, spec.environment)
    if status == "READY":
        sink.write(f"Backend family environment: READY ({spec.environment}).\n")
        return True
    sink.write(f"Backend family environment: {status}. Installing {spec.environment}...\n")
    environment = install_family_environment(spec.environment, root)
    sink.write(f"The isolated {spec.environment} environment is ready at {environment}.\n")
    return True


def _run_inference_action(command: str, source: TextIO, sink: TextIO, root: Path, family: str) -> None:
    """Run serve/demo in the active backend family instead of the core wizard runtime."""
    from reflex.cli import _dispatch_backend_runtime

    dispatched = _dispatch_backend_runtime(command, root, family)
    if dispatched is not None:
        if dispatched != 0:
            sink.write(f"Reflex {command} exited with status {dispatched}.\n")
        return

    # This fallback is used only when the wizard already runs inside the active
    # family runtime (for example during direct developer invocation).
    from reflex.backends.factory import load_service

    service = load_service(root)
    config = load_config(root)
    sink.write("Preparing the model with a fixed private readiness request...\n")
    sink.flush()
    try:
        service.prepare()
    except Exception:
        service.backend.close()
        raise
    sink.write("Model ready.\n")
    sink.flush()
    if command == "demo":
        from reflex.demo import run_demo

        run_demo(service, source, sink, max_line_bytes=config["max_body_bytes"])
        return
    if command == "serve":
        from reflex.server import check_bind_config, serve_service

        check_bind_config(config)
        serve_service(service, config, root)
        return
    raise ValueError(f"Unsupported inference action: {command}")


def _model_manager(source: TextIO, sink: TextIO, root: Path) -> None:
    registry = load_registry(root)
    manager = ModelManager(root, registry)
    while True:
        sink.write("\nMODEL MANAGER\n")
        sink.write(f"  {'#':>2}  {'MODEL':<14} {'STATUS':<14} {'SUPPORT':<12} {'DEVICES':<9} QUALIFIED\n")
        for index, spec in enumerate(registry.values(), 1):
            info = manager.info(spec.model_id, spec.default_variant)
            qualified = ",".join(info.get("qualified_devices", [])) or "none"
            sink.write(
                f"  {index:>2}  {spec.model_id:<14} {info['status']:<14} "
                f"{info['support']:<12} {_device_label(spec):<9} {qualified}\n"
            )
            if spec.variants:
                sink.write(f"       {'VARIANT':<18} {'STATUS':<14} {'SUPPORT':<12} QUALIFIED\n")
            for variant_index, variant_name in enumerate(spec.variants, 1):
                variant_info = manager.info(spec.model_id, variant_name)
                variant_devices = ",".join(variant_info.get("qualified_devices", [])) or "none"
                marker = " (active)" if (
                    spec.model_id == "laya"
                    and load_config(root).get("active_model") == "laya"
                    and load_config(root).get("laya_variant") == variant_name
                ) else ""
                sink.write(
                    f"       {variant_index:>2} {variant_name:<18} {variant_info['status']:<14} "
                    f"{variant_info['support']:<12} {variant_devices}{marker}\n"
                )
        selection = _read(source, sink, "Model number, or B to return: ")
        if selection.casefold() == "b":
            return
        models = list(registry.values())
        model_index = _menu_index(selection, len(models))
        if model_index is None:
            sink.write("Choose one of the listed model numbers.\n")
            continue
        spec = models[model_index]
        variant = spec.default_variant
        if spec.variants:
            sink.write("Checkpoint variants (Model Manager is the only place to install or change Laya's active variant):\n")
            sink.write(f"    {'#':>2} {'VARIANT':<18} {'STATUS':<14} {'SUPPORT':<12} QUALIFIED\n")
            for index, item in enumerate(spec.variants, 1):
                item_info = manager.info(spec.model_id, item)
                qualified_for = ",".join(item_info.get("qualified_devices", [])) or "none"
                default_marker = " (default)" if item == spec.default_variant else ""
                sink.write(
                    f"    {index:>2} {item:<18} {item_info['status']:<14} "
                    f"{item_info['support']:<12} {qualified_for}{default_marker}\n"
                )
            choice = _read(source, sink, "Variant number, or Enter for default: ")
            if choice:
                variant_index = _menu_index(choice, len(spec.variants))
                if variant_index is None:
                    sink.write("No model action started; variant selection was invalid.\n")
                    continue
                variant = spec.variants[variant_index]
        current = manager.info(spec.model_id, variant)
        sink.write(f"{spec.model_id} revision {spec.revision}; disk estimate {spec.estimated_bytes[variant or 'default']:,} bytes.\n")
        if spec.status != "SUPPORTED":
            sink.write("This checkpoint is experimental and cannot be activated or served until its qualification passes.\n")
        elif spec.support_status(variant if spec.model_id == "laya" else None) != "SUPPORTED":
            sink.write("This variant is experimental and cannot be activated for normal inference.\n")
        elif current.get("qualified") is not True:
            from reflex.qualification import qualification_evidence_exists

            if qualification_evidence_exists(root, spec, variant):
                sink.write("This checkpoint has a passing qualification, but its isolated environment no longer matches the qualification lock; Doctor / Repair must fix it before inference.\n")
            else:
                sink.write("This checkpoint variant has no passing qualification and is not runnable for normal inference.\n")
        sink.write("1 Info  2 Download  3 Verify  4 Repair  5 Delete  6 Delete + re-download  7 Set active  8 Install family environment  B Back\n")
        action = _read(source, sink, "Action: ")
        try:
            if action == "1":
                sink.write(f"Status: {current['status']}; support: {current['support']}; source: {spec.source_url}\n")
                sink.write(f"Execution devices: {', '.join(spec.devices)}; qualified devices: {', '.join(current.get('qualified_devices', [])) or 'none'}\n")
                sink.write(f"Required files: {', '.join(spec.required_files)}\n")
            elif action == "2":
                manager.download(spec.model_id, variant)
                sink.write("Pinned checkpoint downloaded and verified.\n")
            elif action == "3":
                sink.write("Verified.\n" if manager.verify(spec.model_id, variant) else "Verification failed.\n")
            elif action == "4":
                manager.repair(spec.model_id, variant)
                sink.write("Repair completed and verified.\n")
            elif action == "5":
                confirmation = _read(source, sink, f"Type DELETE {spec.model_id} to remove this model: ")
                if confirmation == f"DELETE {spec.model_id}":
                    manager.delete(spec.model_id, variant)
                    sink.write("Model files deleted.\n")
                else:
                    sink.write("No changes made.\n")
            elif action == "6":
                manager.download(spec.model_id, variant)
                sink.write("Replacement verified; any previous installation was preserved until promotion.\n")
            elif action == "7":
                updated = manager.select_active(spec.model_id, load_config(root), variant)
                sink.write(f"Active model set to {updated['active_model']}. Restart the server to load it.\n")
            elif action == "8":
                _ensure_family_environment(spec, sink, root)
            elif action.casefold() == "b":
                return
            else:
                sink.write("Choose one of the listed actions.\n")
        except Exception as exc:
            sink.write(f"Model operation failed: {exc}\n")


def _setup_screen(source: TextIO, sink: TextIO, root: Path) -> None:
    registry = load_registry(root)
    manager = ModelManager(root, registry)
    sink.write("\nSETUP\n1 Quick Setup (Laya multilingual)\n2 Custom Setup\nB Back\n")
    mode = _read(source, sink, "Select: ")
    if mode.casefold() == "b":
        return
    if mode == "1":
        model_id, variant = "laya", "multilingual"
    elif mode == "2":
        sink.write("Curated models:\n")
        models = list(registry.values())
        sink.write(f"  {'#':>2}  {'MODEL':<14} {'SUPPORT':<13} DEVICES\n")
        for index, spec in enumerate(models, 1):
            sink.write(f"  {index:>2}  {spec.model_id:<14} {spec.status:<13} {_device_label(spec)}\n")
        model_selection = _read(source, sink, "Model number: ")
        model_index = _menu_index(model_selection, len(models))
        if model_index is None:
            sink.write("No setup started; choose a listed model.\n")
            return
        spec = models[model_index]
        model_id, variant = spec.model_id, spec.default_variant
        if spec.model_id == "laya":
            sink.write("Custom Setup uses the default multilingual checkpoint. Install or change other Laya variants later in Model Manager.\n")
    else:
        sink.write("Choose Quick Setup, Custom Setup, or Back.\n")
        return

    spec = registry[model_id]
    sink.write(f"\n{model_id} uses pinned revision {spec.revision}; release support status is {spec.status}.\n")
    if spec.status != "SUPPORTED":
        sink.write("This model is experimental and cannot be selected for normal server or Demo inference yet.\n")

    # Setup owns runtime readiness independently from checkpoint download. A
    # user may postpone multi-GB model files and still leave the backend family
    # environment ready for a later Model Manager download.
    try:
        if not _ensure_family_environment(spec, sink, root):
            sink.write("Setup stopped before model download because the backend environment is not ready.\n")
            return
    except Exception as exc:
        sink.write(f"Backend environment setup failed: {exc}\n")
        return

    from reflex.qualification import qualification_evidence_exists, qualified_devices

    config = load_config(root)
    selected_policy = config.get("device_policy", "auto")
    checkpoint_variant = variant if model_id == "laya" else None
    passing_devices = qualified_devices(root, spec, checkpoint_variant)
    qualified = bool(passing_devices) if selected_policy == "auto" else selected_policy in passing_devices
    if spec.status == "SUPPORTED" and not qualified and any(
        qualification_evidence_exists(root, spec, checkpoint_variant, device)
        for device in spec.devices
    ):
        sink.write("Warning: the installed backend environment does not match the release qualification lock; Doctor / Repair must fix it before inference.\n")

    answer = _read(source, sink, "Download and verify the model now? [y/N]: ")
    if answer.casefold() not in {"y", "yes"}:
        sink.write("Backend environment is ready; model download was postponed to Model Manager.\n")
        return

    manager.download(model_id, variant)
    if spec.status == "SUPPORTED" and qualified:
        manager.select_active(model_id, load_config(root), variant)
    else:
        sink.write("The pinned files are installed and verified, but this model variant remains unavailable until qualification/runtime validation passes.\n")


def _doctor_screen(source: TextIO, sink: TextIO, root: Path) -> None:
    sink.write("\nDOCTOR\n1 Fast Doctor\n2 Extended Doctor (loads the active model and runs a local golden request)\nB Back\n")
    mode = _read(source, sink, "Select [1]: ")
    if mode.casefold() in {"b", "back"}:
        return
    if mode not in {"", "1", "2"}:
        sink.write("Choose Fast Doctor, Extended Doctor, or Back.\n")
        return
    extended = mode == "2"
    if extended:
        sink.write("Extended Doctor may use GPU memory and take several minutes; it will not start an HTTP listener.\n")
    checks = inspect(root, extended=extended)
    for check in checks:
        sink.write(f"{check['status']:<15} {check['name']}: {check['message']}\n")
    proposals = repair_preview(checks)
    if not proposals:
        return
    sink.write("Repair preview:\n")
    for index, proposal in enumerate(proposals, 1):
        sink.write(f"{index}. [{proposal['action_id']}] {proposal['component']}: {proposal['action']}\n")
    selected = _read(source, sink, "Enter a repair number to apply it, or Enter to return: ")
    if not selected:
        return
    try:
        selection = int(selected)
        if not 1 <= selection <= len(proposals):
            raise ValueError
        proposal = proposals[selection - 1]
    except (ValueError, IndexError):
        sink.write("No repair started; selection was invalid.\n")
        return
    if proposal["component"] == "active_model_files":
        config = load_config(root)
        model_id = config["active_model"]
        variant = config["laya_variant"] if model_id == "laya" else None
        ModelManager(root).repair(model_id, variant)
        sink.write("The selected model is verified.\n")
    elif proposal["component"] == "portable_python":
        sink.write("Restart the Reflex launcher to run the checksum-verified portable runtime bootstrap.\n")
    elif proposal["component"] == "configuration":
        repaired = repair_configuration(root)
        sink.write(f"Configuration reset to safe defaults; selected model is {repaired['active_model']}. A broken file was retained as a dated backup.\n")
    elif proposal["component"].startswith("environment_"):
        family = proposal["component"].removeprefix("environment_")
        environment = install_family_environment(family, root)
        sink.write(f"The isolated {family} environment was rebuilt at {environment}.\n")
    else:
        sink.write("No automatic repair is available for this component; the preview identifies the smallest affected layer.\n")


def _cuda_device_status(root: Path, spec) -> tuple[bool, str]:
    """Check CUDA from this model's isolated runtime, including static model floors."""
    executable = root / "runtime" / "envs" / spec.environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not executable.is_file():
        return False, f"The isolated {spec.environment} environment is not installed."
    requirements = spec.runtime_requirements
    minimum = requirements.minimum_free_vram_bytes if requirements else 0
    capability = requirements.minimum_compute_capability if requirements else (0, 0)
    probe = f'''import json, torch
available = bool(torch.cuda.is_available())
reason = "CUDA is not available in this backend environment"
if available:
    props = torch.cuda.get_device_properties(0)
    free, _total = torch.cuda.mem_get_info(0)
    cap = (props.major, props.minor)
    floor = ({capability[0]}, {capability[1]})
    minimum = {minimum}
    reason = (
        "GPU compute capability is below the model requirement" if cap < floor else
        "GPU does not support BF16" if not torch.cuda.is_bf16_supported() else
        "GPU has insufficient free VRAM for the model" if free < minimum else
        ""
    )
print(json.dumps({{"available": available and not bool(reason), "reason": reason}}))
'''
    try:
        result = subprocess.run([str(executable), "-c", probe], capture_output=True, text=True, timeout=20, check=False)
        payload = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        payload = {}
    if not isinstance(payload, dict) or payload.get("available") is not True:
        reason = payload.get("reason") if isinstance(payload, dict) else None
        return False, str(reason or "CUDA could not be verified in the active backend environment.")
    return True, "CUDA is available and meets the registry's device requirements."


def _device_policy_reason(root: Path, spec, device: str, qualified: set[str], *, environment_ready: bool | None = None) -> str | None:
    if device == "auto":
        candidates = []
        if "cuda" in spec.devices:
            candidates.append(_device_policy_reason(root, spec, "cuda", qualified, environment_ready=environment_ready))
        if "cpu" in spec.devices:
            candidates.append(_device_policy_reason(root, spec, "cpu", qualified, environment_ready=environment_ready))
        return None if any(reason is None for reason in candidates) else "No declared execution device has a passing qualification and ready runtime."
    if device not in spec.devices:
        return f"{spec.model_id} does not support {device.upper()}."
    if device not in qualified:
        return f"No passing {device.upper()} qualification exists for this model and variant."
    if environment_ready is None:
        environment_ready = _family_runtime_status(root, spec.environment) == "READY"
    if not environment_ready:
        return f"The isolated {spec.environment} environment is not READY; run Setup or Doctor / Repair."
    if device == "cuda":
        ready, reason = _cuda_device_status(root, spec)
        return None if ready else reason
    return None


def _settings(source: TextIO, sink: TextIO, root: Path) -> None:
    import ipaddress
    import secrets

    while True:
        config = load_config(root)
        active = load_registry(root)[config["active_model"]]
        variant = config["laya_variant"] if active.model_id == "laya" else None
        from reflex.qualification import qualified_devices

        qualified = set(qualified_devices(root, active, variant))
        environment_ready = _family_runtime_status(root, active.environment) == "READY"
        sink.write("\nSETTINGS\n")
        sink.write(f"Active model: {active.model_id}\n")
        if variant:
            sink.write(f"Active Laya variant: {variant} ({active.support_status(variant)})\n")
        sink.write(f"Run models on: {config['device_policy'].upper()}\n")
        sink.write(f"Qualified devices: {', '.join(device.upper() for device in sorted(qualified)) or 'none'}\n")
        sink.write(f"HTTP address: {config['host']}:{config['port']}\n")
        sink.write(f"Log verbosity: {config['log_level']}\n")
        sink.write("\n1 Run models on CPU or NVIDIA CUDA\n2 Change log verbosity\n3 Change bind address / port\nB Back\n")
        action = _read(source, sink, "Choice: ")
        if action.casefold() == "b":
            return
        if action == "1":
            options = (("auto", "Auto — choose a qualified compatible device"),
                       ("cpu", "CPU"), ("cuda", "NVIDIA CUDA"))
            availability = {device: _device_policy_reason(root, active, device, qualified,
                                                          environment_ready=environment_ready)
                            for device, _label in options}
            sink.write("\nRUN DEVICE\n")
            for index, (device, label) in enumerate(options, 1):
                reason = availability[device]
                detail = "AVAILABLE" if reason is None else f"NOT AVAILABLE — {reason}"
                current = " (current)" if device == config["device_policy"] else ""
                sink.write(f"{index}. {label}: {detail}{current}\n")
            selected = _read(source, sink, "Choice number, or B to return: ")
            if selected.casefold() == "b":
                continue
            if selected not in {"1", "2", "3"}:
                sink.write("No setting changed; choose a listed number.\n")
                continue
            policy = options[int(selected) - 1][0]
            reason = availability[policy]
            if reason is not None:
                sink.write(f"No setting changed; {reason}\n")
                continue
            config["device_policy"] = policy
            save_config(config, root)
            sink.write(f"Execution policy saved as {policy.upper()}; restart the server or Demo Mode to apply it.\n")
        elif action == "2":
            sink.write("\nLOG VERBOSITY\n1 QUIET — warnings and errors\n2 NORMAL — standard status messages\n3 DEBUG — detailed diagnostics\nB Back\n")
            selected = _read(source, sink, "Choice number: ")
            levels = {"1": "QUIET", "2": "NORMAL", "3": "DEBUG"}
            if selected.casefold() == "b":
                continue
            if selected not in levels:
                sink.write("No setting changed; choose a listed number.\n")
                continue
            config["log_level"] = levels[selected]
            save_config(config, root)
            sink.write(f"Log verbosity saved as {levels[selected]}.\n")
        elif action == "3":
            host = _read(source, sink, f"Bind host [{config['host']}]: ") or config["host"]
            try:
                port_text = _read(source, sink, f"Port [{config['port']}]: ")
                port = int(port_text) if port_text else config["port"]
                ipaddress.ip_address(host)
                if not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                sink.write("No setting changed; use an IPv4/IPv6 address and a port from 1 to 65535.\n")
                continue
            config["host"], config["port"] = host, port
            if host not in {"127.0.0.1", "::1"} and not config.get("api_key"):
                config["api_key"] = secrets.token_urlsafe(32)
                sink.write(f"Generated bearer key (save it now): {config['api_key']}\n")
            save_config(config, root)
            sink.write("Settings saved. Restart the server for network changes.\n")
        else:
            sink.write("Choose one of the listed options.\n")


def _reinstall_screen(source: TextIO, sink: TextIO, root: Path) -> bool:
    sink.write("1 Reinstall runtime and environments; keep models/settings\n")
    sink.write("2 Reinstall runtime and environments; reset settings, keep models\n")
    sink.write("3 Reinstall Laya environment only\n4 Reinstall Decider environment only\n")
    sink.write("5 Reinstall native NVIDIA CUDA environment only\n6 Full reinstall; delete all models\nB Back\n")
    choice = _read(source, sink, "Scope: ")
    scopes = {
        "1": "runtime",
        "2": "reset-settings",
        "3": "family:laya",
        "4": "family:decider",
        "5": "family:decision-cuda",
        "6": "full",
    }
    if choice.casefold() == "b" or choice not in scopes:
        sink.write("No changes made.\n")
        return True
    scope = scopes[choice]
    required = "DELETE ALL MODELS" if scope == "full" else "REINSTALL"
    confirmation = _read(source, sink, f"Type {required} to continue: ")
    if confirmation != required:
        sink.write("No changes made.\n")
        return True
    try:
        completed = clean_reinstall(scope, root=root, confirmation=confirmation)
        if not completed:
            sink.write("Reinstall scheduled. Reflex will close, finish removing the active runtime, and restart from reflex.bat.\n")
            return False
        sink.write("Reinstall scope completed. Models/settings were preserved or removed as selected.\n")
    except ReinstallError as exc:
        sink.write(f"Reinstall failed: {exc}\n")
    return True


def run_wizard(source: TextIO = sys.stdin, sink: TextIO = sys.stdout, root: Path | None = None) -> None:
    root = (root or repository_root()).resolve()
    branding_shown = os.environ.get("REFLEX_BANNER_SHOWN") == "1"
    while True:
        try:
            config_broken = False
            try:
                config = load_config(root)
            except Exception:
                config = dict(DEFAULT_CONFIG)
                config_broken = True
            registry = load_registry(root)
            active = registry[config["active_model"]]
            variant = config["laya_variant"] if active.model_id == "laya" else active.default_variant
            model_status = ModelManager(root, registry).info(active.model_id, variant)["status"]
            runtime_status = _family_runtime_status(root, active.environment)
            render_main_menu(
                sink,
                model=active.model_id,
                support=active.support_status(variant if active.model_id == "laya" else None),
                variant=variant or "-",
                runtime=runtime_status,
                model_files=model_status,
                config_broken=config_broken,
                show_branding=not branding_shown,
            )
            branding_shown = True
            choice = _read(source, sink, "  Select: ")
        except (EOFError, KeyboardInterrupt):
            sink.write("\n")
            return
        if choice == "0":
            return
        try:
            if choice == "1":
                _setup_screen(source, sink, root)
            elif choice == "2":
                _model_manager(source, sink, root)
            elif choice == "3":
                _doctor_screen(source, sink, root)
            elif choice == "4":
                _settings(source, sink, root)
            elif choice in {"5", "6"}:
                command = "serve" if choice == "5" else "demo"
                if command == "demo":
                    sink.write(
                        f"\nDEMO MODE\nActive model: {active.model_id}\n"
                        "Paste one-line System One JSON. Type EXIT to return to the menu.\n"
                    )
                    sink.flush()
                _run_inference_action(command, source, sink, root, active.environment)
            elif choice == "7":
                if not _reinstall_screen(source, sink, root):
                    return
            else:
                sink.write("Choose one of the listed options.\n")
        except (EOFError, KeyboardInterrupt):
            sink.write("\n")
            return
        except Exception as exc:
            sink.write(f"Action failed: {exc}\n")

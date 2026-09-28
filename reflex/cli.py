"""Reflex command line."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from reflex.config import load_config
from reflex.doctor import inspect
from reflex.errors import ReflexError
from reflex.paths import repository_root
from reflex.registry import load_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reflex")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("wizard")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--extended", action="store_true", help="Load the active model and run a local golden request.")
    sub.add_parser("serve")
    sub.add_parser("demo")
    sub.add_parser("models")
    return parser


def _configure_stdio_encoding() -> None:
    """Keep CLI JSON and localized console text usable on Windows code pages."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _dispatch_backend_runtime(command: str, root: Path, family: str) -> int | None:
    """Re-exec inference in the selected, repository-owned backend runtime."""
    if command not in {"serve", "demo"}:
        return None
    runtime = root / "runtime"
    executable = runtime / "envs" / family / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        raise ReflexError("environment_missing", f"The isolated {family} backend environment is not installed. Run Setup and Doctor.", 503)
    try:
        same_runtime = os.path.samefile(sys.executable, executable)
    except OSError:
        same_runtime = os.path.normcase(os.path.realpath(sys.executable)) == os.path.normcase(os.path.realpath(executable))
    if same_runtime:
        return None
    env = dict(os.environ)
    env["REFLEX_ROOT"] = str(root)
    try:
        completed = subprocess.run(
            [str(executable), "-m", "reflex", command],
            cwd=root,
            env=env,
            check=False,
        )
    except OSError:
        raise ReflexError("environment_missing", f"The isolated {family} backend environment could not be started. Repair that environment.", 503) from None
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_encoding()
    args = _parser().parse_args(argv)
    command = args.command or "wizard"
    root = repository_root()
    try:
        if command == "wizard":
            from reflex.wizard import run_wizard
            run_wizard(root=root)
            return 0
        if command == "doctor":
            print(json.dumps(inspect(root, extended=args.extended), ensure_ascii=False, indent=2))
            return 0
        registry = load_registry(root)
        config = load_config(root)
        if command == "models":
            print(json.dumps([
                {"id": spec.model_id, "status": spec.status, "revision_pinned": spec.pinned}
                for spec in registry.values()
            ], indent=2))
            return 0
        spec = registry[config["active_model"]]
        dispatched = _dispatch_backend_runtime(command, root, spec.environment)
        if dispatched is not None:
            return dispatched
        if command == "serve":
            from reflex.server import check_bind_config, serve_service
            check_bind_config(config)
        from reflex.backends.factory import load_service
        print(f"Loading the selected {config['active_model']} model...", file=sys.stderr, flush=True)
        service = load_service(root)
        if command == "demo":
            from reflex.demo import run_demo
            print("Preparing the model with a fixed private readiness request...", file=sys.stderr, flush=True)
            try:
                service.prepare()
            except Exception:
                service.backend.close()
                raise
            print("Model ready.", file=sys.stderr, flush=True)
            run_demo(service, sys.stdin, sys.stdout, max_line_bytes=config["max_body_bytes"])
            return 0
        if command == "serve":
            serve_service(service, config, root)
            return 0
    except ReflexError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Reflex could not continue: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

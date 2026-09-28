"""Console branding shared by the Reflex wizard and Windows bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO


_RESOURCE = Path(__file__).resolve().parents[1] / "assets" / "reflex-branding.txt"
_WORDMARK_MARKER = "--- REFLEX_WORDMARK ---"
_TAGLINE_MARKER = "--- REFLEX_TAGLINE ---"


def _read_branding() -> tuple[str, str]:
    lines = _RESOURCE.read_text(encoding="utf-8").splitlines()
    try:
        wordmark_start = lines.index(_WORDMARK_MARKER) + 1
        tagline_start = lines.index(_TAGLINE_MARKER)
    except ValueError as exc:
        raise RuntimeError("The shared Reflex branding resource is invalid.") from exc
    if tagline_start <= wordmark_start:
        raise RuntimeError("The shared Reflex branding resource is invalid.")
    wordmark = "\n".join(lines[wordmark_start:tagline_start]).rstrip()
    tagline = "\n".join(lines[tagline_start + 1:]).rstrip()
    if not wordmark or not tagline:
        raise RuntimeError("The shared Reflex branding resource is incomplete.")
    return wordmark, tagline


REFLEX_WORDMARK, REFLEX_TAGLINE = _read_branding()

_MENU = (
    ("1", "Quick / Custom Setup"),
    ("2", "Model Manager"),
    ("3", "Doctor / Repair"),
    ("4", "Settings"),
    ("5", "Start HTTP server"),
    ("6", "Demo Mode"),
    ("7", "Clean Reinstall"),
    ("0", "Exit"),
)

_STATUS_BOX_WIDTH = 56
_STATUS_LABEL_WIDTH = 12
_STATUS_VALUE_WIDTH = 39
_STATUS_HEADER = "─ STATUS "
_STATUS_TOP = "    ╭" + _STATUS_HEADER + "─" * (_STATUS_BOX_WIDTH - len(_STATUS_HEADER) - 2) + "╮\n"
_STATUS_BOTTOM = "    ╰" + "─" * (_STATUS_BOX_WIDTH - 2) + "╯\n"


def _status_line(label: str, value: str) -> str:
    return f"    │ {label:<{_STATUS_LABEL_WIDTH}} {value:<{_STATUS_VALUE_WIDTH}} │\n"


def render_main_menu(
    sink: TextIO,
    *,
    model: str,
    support: str,
    variant: str,
    runtime: str,
    model_files: str,
    config_broken: bool = False,
    show_branding: bool = True,
) -> None:
    """Render the stable Reflex home screen without terminal-specific escape codes."""
    sink.write("\n")
    if show_branding:
        sink.write(REFLEX_WORDMARK)
        sink.write("\n\n")
        sink.write(REFLEX_TAGLINE)
        sink.write("\n\n")

    if config_broken:
        sink.write("  ! Configuration is BROKEN; Doctor / Repair can reset it while keeping model files.\n\n")

    sink.write(_STATUS_TOP)
    sink.write(_status_line("Model", model))
    sink.write(_status_line("Support", support))
    sink.write(_status_line("Variant", variant))
    sink.write(_status_line("Runtime", runtime))
    sink.write(_status_line("Model files", model_files))
    sink.write(_STATUS_BOTTOM)
    sink.write("\n")

    for key, label in _MENU:
        sink.write(f"  {key}  {label}\n")
    sink.write("\n")

"""Jinja2-based rendering of template files using experiment config values."""

from __future__ import annotations

import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError


# File extensions treated as text and eligible for Jinja2 rendering.
# .j2 files are always rendered; the .j2 suffix is stripped from the output name.
_TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".j2", ".txt", ".md", ".sh", ".cfg", ".ini", ".toml",
}

# Filenames that are never rendered (kept as-is)
_SKIP_RENDER = {".gitkeep"}


def render_template_dir(
    template_dir: Path,
    dest_dir: Path,
    context: dict,
) -> None:
    """Copy *template_dir* to *dest_dir*, rendering every eligible text file
    through Jinja2 with *context*.

    Args:
        template_dir: Source template directory.
        dest_dir:     Destination experiment directory (will be created).
        context:      Dictionary of values injected into Jinja2 templates.

    Raises:
        FileExistsError: If *dest_dir* already exists.
        TemplateError:   If a Jinja2 template contains an undefined variable.
    """
    if dest_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {dest_dir}\n"
            "Use a different experiment name or remove the directory first."
        )

    dest_dir.mkdir(parents=True)

    for src_path in template_dir.rglob("*"):
        rel = src_path.relative_to(template_dir)
        dst_path = dest_dir / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        # Strip .j2 suffix from the destination path
        if dst_path.suffix == ".j2":
            dst_path = dst_path.with_suffix("")

        if src_path.name in _SKIP_RENDER or src_path.suffix not in _TEXT_SUFFIXES:
            shutil.copy2(src_path, dst_path)
            continue

        _render_file(src_path, dst_path, template_dir, context)


def _render_file(
    src_path: Path,
    dst_path: Path,
    template_root: Path,
    context: dict,
) -> None:
    """Render a single Jinja2 template file and write to *dst_path*."""
    env = Environment(
        loader=FileSystemLoader(str(template_root)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )

    rel_str = str(src_path.relative_to(template_root))
    try:
        template = env.get_template(rel_str)
        rendered = template.render(**context)
    except TemplateError as exc:
        raise TemplateError(
            f"Failed to render template '{rel_str}': {exc}"
        ) from exc

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(rendered, encoding="utf-8")

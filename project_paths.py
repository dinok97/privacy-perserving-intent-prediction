"""Project locations and compatibility with saved pre-reorganisation metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_project_path(value, *, base=None):
    """Resolve a stored path, relocating the former outer video_processing folder.

    Existing paths and external installations are preserved. This lets old local
    manifests and depth metadata work without rewriting historical provenance.
    """
    path = Path(value).expanduser()
    candidate = (Path(base) / path if base is not None and not path.is_absolute()
                 else ROOT / path if not path.is_absolute() else path).resolve()
    if candidate.exists():
        return candidate
    try:
        suffix = candidate.relative_to(ROOT / 'video_processing')
    except ValueError:
        if path.is_absolute() or not path.parts or path.parts[0] != 'video_processing':
            return candidate
        suffix = Path(*path.parts[1:])
    return ROOT / suffix

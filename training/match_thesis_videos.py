"""Match feature exports to the source videos recorded in the MasterData logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


MASTERDATA_FOLDERS = {"masterdata_enter": "enter", "masterdata_pass": "pass"}
SOURCE_INPUT = re.compile(
    r"\|\s*pipeline\s*\|\s*input:\s+.*[\\/]MasterData[\\/]"
    r"(enter|pass)[\\/]([^\\/]+\.(?:mp4|mov|avi|mkv|m4v))\s*$",
    re.IGNORECASE,
)
TRACK_SUFFIX = re.compile(r"(_(?:enter|pass|exit))_\d+$")


def source_video_key(folder: str, clip_stem: str) -> tuple[str, str]:
    """Use the source folder label and original stem, regardless of tracker label."""
    return MASTERDATA_FOLDERS.get(folder, folder), TRACK_SUFFIX.sub(r"\1", clip_stem)


def read_source_videos(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        (clip["source_label"], Path(clip["source_filename"]).stem)
        for clip in manifest["clips"]
    }


def build_manifest(data: Path, logs: list[Path]) -> dict:
    clips = {}
    for log in logs:
        with log.open(encoding="utf-8-sig") as stream:
            for line in stream:
                match = SOURCE_INPUT.search(line)
                if not match:
                    continue
                label, filename = match.groups()
                label = label.lower()
                key = (label, Path(filename).stem)
                clip = clips.setdefault(key, {
                    "source_label": label,
                    "source_filename": filename,
                    "source_logs": set(),
                    "feature_exports": [],
                })
                clip["source_logs"].add(log.name)
    if not clips:
        raise ValueError("No original MasterData video inputs found in the supplied logs.")

    for folder in MASTERDATA_FOLDERS:
        for export in sorted((data / folder).rglob("*_features.json")):
            info = json.loads(export.read_text(encoding="utf-8"))["info"]
            stem = info.get("clip") or export.stem.removesuffix("_features")
            clip = clips.get(source_video_key(folder, stem))
            if clip is not None:
                clip["feature_exports"].append(export.relative_to(data).as_posix())

    rows = []
    for key in sorted(clips):
        clip = clips[key]
        rows.append({**clip, "source_logs": sorted(clip["source_logs"])})
    return {"clips": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root of the MasterData exports")
    parser.add_argument("--output", type=Path, required=True, help="Path for the local match manifest")
    parser.add_argument("--log", type=Path, action="append", help="Source log to use (repeatable)")
    args = parser.parse_args()
    logs = args.log or sorted(
        log for folder in MASTERDATA_FOLDERS
        for log in (args.data / folder / "logs").glob("pipeline_*.log")
    )
    manifest = build_manifest(args.data, logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    clips = manifest["clips"]
    available = sum(bool(clip["feature_exports"]) for clip in clips)
    print(f"Matched {len(clips)} source videos. {available} have feature exports.")


if __name__ == "__main__":
    main()

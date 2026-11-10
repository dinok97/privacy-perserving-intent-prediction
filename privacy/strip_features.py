"""
strip_features.py  -  privacy-reduced copy of a <clip>_features.json (schema feat-0.3)

TEMPORARY post-processing step. It reads a finished features file, removes the
fields listed in privacy/README.md for the chosen tier and writes a NEW file.
The input is never modified. See privacy/README.md for the reasoning and the
list of fields; the lists below are the single source of truth for this script.

Tiers (cumulative)
    F0   nothing removed (baseline for the ablation)
    F1   size:        absolute body size and the pixel twins that reveal it   (7 per-frame + 3 track-level)
    F2   F1 + body proportions                                                (+3 per-frame)
    F3   F2 + gait cues                                                       (+2 per-frame)

Besides the tiers, for F1-F3 the script also
    * pseudonymizes metadata that links the file to the raw video, a recording
      date/time or a file name (clip, clip_video, clip_video_sha256, session_id,
      source file names)                                  [--metadata off to disable]
    * removes the 3D "estimated metre" fields and the metric calibration, because
      metres divided into any *_bh feature give the body height in metres
                                                          [--keep-3d-metres to keep]

Usage
    python privacy/strip_features.py --tier F2 --input data/pipeline/<clip>_features.json
    python privacy/strip_features.py --tier F3 --input-dir data/pipeline --out-dir data/privacy_F3
    python privacy/strip_features.py --tier F1 --input-dir data/pipeline --catalog data/pipeline/feature_catalog.json
    python privacy/strip_features.py --tier F2 --input x_features.json --key-file secret.key   # adds info.clip_id

Output is named <stem>_features_<tier>.json (next to the input unless --out-dir is given).
A file is only written if a final audit finds none of the removed field names left.
This is data minimisation, not a proof of anonymity: no re-identification test was run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path

TIERS = ("F0", "F1", "F2", "F3")

# ---- per-frame fields: paths are relative to one entry of data.frames.
#      "name[]" = every element of a list (null entries are skipped).
FRAME_PATHS = {
    "F1": [
        "pos.bbox_h_px",                       # body height in px (bbox)
        "pos.bh_ref_px",                       # body-height reference used for all _bh units
        "pose.body_height_px",                 # nose to mid-ankle, px
        "kin.speed_px_s",                      # twin of speed_bh_s
        "kin.curvature_per_px",                # = turn rate / px speed -> reveals the px speed, hence bh
        "scene_rel.dist_block_px",             # twin of dist_block_bh
        "scene_rel.target.dist_px",            # twin of target.dist_bh
        "scene_rel.competitors[].dist_px",     # same block, per competitor
        "scene_rel.exits[].dist_px",           # same block, per exit
    ],
    "F2": [
        "pose.shoulder_width_bh",
        "pose.hip_width_bh",
        "pose.leg_length_bh",
    ],
    "F3": [
        "pose.ankle_sep_bh",
        "pose.wrist_spread_bh",
    ],
}

# ---- track-level fields: paths are relative to the "data" object.
DATA_PATHS = {
    "F1": [
        "track_summary.causal.first_bh_px",
        "track_summary.hindsight.camera_geom.bh_vs_y_slope",       # scales with person height
        "track_summary.hindsight.camera_geom.bh_vs_y_intercept",
    ],
    "F2": [],
    "F3": [],
}
# camera_geom.horizon_y_px (= -intercept/slope) is a property of the camera and is kept.

# ---- 3D extension (only present with --with-3d)
THREE_D_FRAME_PATHS = [
    "geometry_3d.point_ground_estimated_m",
    "geometry_3d.distances_estimated_m",
    "geometry_3d.closure_rates_estimated_m_per_s",
]
THREE_D_INFO_PATHS = ["depth.metric_calibration"]       # scale factor would undo the removal above

# ---- metadata (info.*) that links the file to the raw video, a date/time or a file name
INFO_DROP_PATHS = [
    "clip", "clip_video", "clip_video_sha256",
    "trajectory_meta.session_id",                      # = recording date + clock time
    "depth.source_video_sha256",
]
# info.sources holds file names derived from the clip name -> replaced by true/false (was the input used)

# Names that must not appear anywhere in the output after stripping.
# dist_px is allowed under origin_exit (distance to the exit at first detection; no _bh twin in the output).
ALLOWED_UNDER = {"dist_px": "origin_exit"}


def _tier_upto(tier):
    return TIERS[1:TIERS.index(tier) + 1] if tier != "F0" else ()


def paths_for(tier):
    frame = [p for t in _tier_upto(tier) for p in FRAME_PATHS[t]]
    data = [p for t in _tier_upto(tier) for p in DATA_PATHS[t]]
    return frame, data


def _delete(obj, parts):
    """Delete the key at `parts` inside obj. Returns the number of keys deleted."""
    if not isinstance(obj, dict) or not parts:
        return 0
    head, rest = parts[0], parts[1:]
    many = head.endswith("[]")
    key = head[:-2] if many else head
    if key not in obj:
        return 0
    if not rest:
        del obj[key]
        return 1
    child = obj[key]
    if many:
        return sum(_delete(item, rest) for item in child) if isinstance(child, list) else 0
    return _delete(child, rest)


def _pseudonym(name, key):
    return "clip_" + hmac.new(key, name.encode("utf-8"), hashlib.sha256).hexdigest()[:12]


def _forbidden_names(frame_paths, data_paths, extra=()):
    names = {p.split(".")[-1] for p in list(frame_paths) + list(data_paths)}
    return names | set(extra)


def audit(result, forbidden, info_forbidden):
    """Return the paths of forbidden keys still present anywhere in the result."""
    left = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                allowed = ALLOWED_UNDER.get(k)
                if k in forbidden and not (allowed and allowed in path):
                    left.append(here)
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(result.get("data", {}), "data")
    info = result.get("info", {})
    for name in info_forbidden:
        node = info
        parts = name.split(".")
        for p in parts[:-1]:
            node = node.get(p, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict) and parts[-1] in node:
            left.append("info." + name)
    walk({k: v for k, v in info.items() if k != "privacy"}, "info")
    return left


def strip(result, tier, *, metadata=True, keep_3d_metres=False, key=None):
    """Return (stripped copy, report). `result` is not modified."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; use one of {TIERS}")
    out = copy.deepcopy(result)
    frame_paths, data_paths = paths_for(tier)
    report = {"tier": tier, "frame_fields": {}, "data_fields": {}, "metadata": [], "three_d": []}

    frames = out.get("data", {}).get("frames", [])
    for p in frame_paths:
        report["frame_fields"][p] = sum(_delete(fr, p.split(".")) for fr in frames)
    for p in data_paths:
        report["data_fields"][p] = _delete(out.get("data", {}), p.split("."))

    drop_3d = tier != "F0" and not keep_3d_metres
    if drop_3d:
        for p in THREE_D_FRAME_PATHS:
            n = sum(_delete(fr, p.split(".")) for fr in frames)
            if n:
                report["three_d"].append((p, n))
        for p in THREE_D_INFO_PATHS:
            if _delete(out.get("info", {}), p.split(".")):
                report["three_d"].append(("info." + p, 1))

    info = out.get("info", {})
    if metadata and tier != "F0":
        original = info.get("clip") or info.get("clip_video")
        for p in INFO_DROP_PATHS:
            if _delete(info, p.split(".")):
                report["metadata"].append("info." + p)
        sources = info.get("sources")
        if isinstance(sources, dict):
            info["sources"] = {k: bool(v) for k, v in sources.items()}
            report["metadata"].append("info.sources (file names -> true/false)")
        if key is not None and original:
            info["clip_id"] = _pseudonym(str(original), key)
            report["metadata"].append("info.clip_id (HMAC pseudonym added)")

    info_forbidden = list(INFO_DROP_PATHS) if (metadata and tier != "F0") else []
    forbidden = _forbidden_names(frame_paths, data_paths,
                                 extra=[p.split(".")[-1] for p in (THREE_D_FRAME_PATHS if drop_3d else [])])
    leftover = audit(out, forbidden, info_forbidden)
    if leftover:
        raise RuntimeError("audit failed, removed names still present: " + ", ".join(leftover[:10]))

    removed = (sorted(frame_paths) + sorted(data_paths)
               + (sorted(THREE_D_FRAME_PATHS) if drop_3d else []))
    out.setdefault("info", {})["privacy"] = {
        "tier": tier,
        "script": "privacy/strip_features.py",
        "removed_fields": removed,
        "metadata_pseudonymized": bool(metadata and tier != "F0"),
        "dropped_3d_metres": bool(drop_3d),
        "note": "Data minimisation based on the literature in privacy/README.md. "
                "Not verified anonymisation: no re-identification test was run.",
    }
    return out, report


def filter_catalog(catalog, tier, keep_3d_metres=False):
    """Remove stripped features from feature_catalog.json content."""
    frame_paths, data_paths = paths_for(tier)
    names = set(frame_paths) | set(data_paths)
    drop_3d = tier != "F0" and not keep_3d_metres
    kept = []
    for entry in catalog.get("features", []):
        n = entry["name"]
        if n in names or (drop_3d and n.startswith("geometry_3d.") and "_estimated_m" in n):
            continue
        kept.append(entry)
    return {**catalog, "features": kept, "privacy_tier": tier}


def _out_path(src, tier, out_dir):
    stem = src.name[:-len(".json")]
    target = Path(out_dir) if out_dir else src.parent
    return target / f"{stem}_{tier}.json"


def _inputs(args):
    if args.input:
        return [Path(args.input)]
    files = sorted(Path(args.input_dir).rglob("*_features.json"))
    return [f for f in files if not re.search(r"_features_F[0-3]\.json$", f.name)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True, choices=TIERS, help="which tier to apply (cumulative)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="one <clip>_features.json")
    src.add_argument("--input-dir", help="every *_features.json below this folder")
    ap.add_argument("--out-dir", help="default: next to each input")
    ap.add_argument("--metadata", choices=("on", "off"), default="on",
                    help="pseudonymize clip names, hashes, session id, source file names (default on for F1-F3)")
    ap.add_argument("--keep-3d-metres", action="store_true",
                    help="keep the *_estimated_m fields and info.depth.metric_calibration (not recommended)")
    ap.add_argument("--key-file", help="file with a secret; adds info.clip_id = HMAC(secret, clip name)")
    ap.add_argument("--catalog", help="feature_catalog.json to filter; written as feature_catalog_<tier>.json")
    ap.add_argument("--dry-run", action="store_true", help="report what would be removed, write nothing")
    args = ap.parse_args(argv)

    key = Path(args.key_file).read_bytes().strip() if args.key_file else None
    files = _inputs(args)
    if not files:
        ap.error("no *_features.json files found")

    failed = 0
    for f in files:
        try:
            result = json.loads(f.read_text(encoding="utf-8"))
            stripped, rep = strip(result, args.tier, metadata=args.metadata == "on",
                                  keep_3d_metres=args.keep_3d_metres, key=key)
        except Exception as error:                                   # keep going; report at the end
            failed += 1
            print(f"FAILED {f}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        removed = sum(rep["frame_fields"].values()) + sum(rep["data_fields"].values())
        removed_3d = sum(n for _, n in rep["three_d"])
        n_frame = len({p.split(".")[-1] for p, n in rep["frame_fields"].items() if n})   # distinct names
        n_track = len([p for p, n in rep["data_fields"].items() if n])
        print(f"{f.name}: {args.tier}  removed {removed} values "
              f"({n_frame} per-frame fields, {n_track} track fields), "
              f"{removed_3d} 3D-metre values, {len(rep['metadata'])} metadata items")
        if args.keep_3d_metres and args.tier != "F0":
            print("  WARNING: --keep-3d-metres: estimated metres / *_bh speed gives body height in metres", file=sys.stderr)
        if args.dry_run:
            continue
        dest = _out_path(f, args.tier, args.out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(stripped, indent=1), encoding="utf-8")
        print(f"  wrote {dest}")

    if args.catalog and not args.dry_run:
        cat_path = Path(args.catalog)
        cat = filter_catalog(json.loads(cat_path.read_text(encoding="utf-8")), args.tier, args.keep_3d_metres)
        dest = Path(args.out_dir or cat_path.parent) / f"feature_catalog_{args.tier}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(cat, indent=1), encoding="utf-8")
        print(f"wrote {dest} ({len(cat['features'])} features)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

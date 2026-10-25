"""
extract_features.py  –  MVP feature extractor (schema feat-0.3)

Inputs per clip (<stem> = <video>_<action>, action = enter/pass/exit):
    <stem>.json              trajectory + COCO-17 keypoints in pixels (Jesper's script)
    <stem>_pose.json         pose sidecar (optional; keypoint fallback only)
    <stem>_background.json   segmentation polygons from background_segmentation.py
    <stem>_background.png    optional; used for brightness features and the debug overlay
    exits_<stem>.json        optional; hand-annotated exits replacing the automatic ones

Output:
    features_<stem>.json     {"info": {...provenance...}, "data": {...features...}}
    feature_catalog.json     one entry per feature (name, level, unit, needs, causal)

Design rules
    * Every per-frame feature is CAUSAL: it uses only the current and past frames,
      so it is safe for online prediction (the door must decide before arrival).
    * Whole-track features live in data.track_summary.hindsight and are flagged.
    * Exits = detected doors + frame-edge gaps the floor reaches + floor tips cut off
      by a wall; the target door is never a competitor. Exit order is only valid inside one file (no stable IDs).
    * Units are in the name: _px pixels, _bh body heights, _s seconds, _deg degrees.

Usage
    python extract_features.py --traj <stem>.json                 (sidecar files found by name)
    python extract_features.py --input-dir data/enter --out-dir data/features --debug
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = "feat-0.3"

# ---------------------------------------------------------------- parameters
PARAMS = {
    "min_poly_area_px": 30,        # drop slivers / 2-point "polygons"
    "kp_conf_min": 0.3,            # keypoint counts as visible above this
    "edge_band_px": 4,             # how far inward an edge pixel looks for a blocker
    "min_gap_px": 15,              # narrower edge gaps are ignored
    "inward_depth_px": 20,         # strip checked behind an edge gap
    "inward_open_frac": 0.6,       # share of that strip that must be unblocked
    "floor_touch_px": 12,          # floor must be this close to an edge gap for it to count
    "tip_min_floor_frac": 0.10,    # floor patches smaller than this share of the largest
                                   # patch never produce exits (stray specks of floor)
    "tip_open_radius_px": 20,      # floor "body" = floor opened with a disk of this radius
    "tip_min_protrusion_px": 10,   # tip must stick out this far beyond the body
    "tip_dir_back_px": 20,         # tip direction measured over this much skeleton
    "tip_look_px": 25,             # look this far beyond the tip for what cuts it off
    "tip_cone_deg": 35,            # ... within this half-angle of the tip direction
    "tip_merge_px": 30,            # tips closer than this are one exit
    "edge_margin_px": 10,          # tips this close to the frame border are edge exits
    "door_bottom_band_px": 15,     # fallback threshold: door points this close to its bottom
    "door_contact_px": 6,          # door outline this close to floor = threshold
    "door_min_contact_pts": 5,     # fewer contact points -> use the fallback
    "origin_max_dist_bh": 1.0,     # first point within this of an exit => origin
    "min_speed_bh_s": 0.1,         # below this, heading/alignment are undefined
    "smooth_window": 3,            # causal moving average on ground point
    "n_competitors": 2,
}

BLOCKING = ("wall", "window")      # things a person cannot walk through
KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
      "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]
K = {n: i for i, n in enumerate(KP)}


# ---------------------------------------------------------------- helpers
def rnd(x, n=3):
    """Round for JSON; NaN/inf/None -> None, numpy -> python."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return None if not math.isfinite(x) else round(x, n)


def ang_diff(a, b):
    """Smallest absolute difference between two angles in degrees (0..180)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def poly_area(poly):
    return abs(cv2.contourArea(np.asarray(poly, dtype=np.float32)))


def seg_relation(p, seg, heading_deg):
    """Distance from p to segment, nearest point, and heading alignment (0 if the heading
    ray points anywhere inside the segment's angular span)."""
    a, b = np.asarray(seg[0], float), np.asarray(seg[1], float)
    ab = b - a
    L2 = float(ab @ ab)
    u = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0, 1))
    near = a + u * ab
    dist = float(np.hypot(*(near - p)))
    align = None
    if heading_deg is not None:
        ta = math.degrees(math.atan2(*(a - p)[::-1]))
        tb = math.degrees(math.atan2(*(b - p)[::-1]))
        span = ang_diff(ta, tb)
        if ang_diff(heading_deg, ta) + ang_diff(heading_deg, tb) <= span + 1e-6:
            align = 0.0
        else:
            align = min(ang_diff(heading_deg, ta), ang_diff(heading_deg, tb))
    return dist, near, align


# ---------------------------------------------------------------- scene
class Scene:
    """Rasterised segmentation + exit detection."""

    def __init__(self, scene_json: dict, W: int, H: int, target_px, manual_exits=None):
        self.manual_exits = manual_exits
        self.W, self.H = W, H
        self.target = np.asarray(target_px, float)
        self.objects = []
        self.dropped = 0
        for o in scene_json.get("objects", []):
            poly = o.get("polygon") or []
            if len(poly) < 3 or poly_area(poly) < PARAMS["min_poly_area_px"]:
                self.dropped += 1
                continue
            self.objects.append(o)

        self.masks = {}
        for o in self.objects:
            lab = o["label"].lower()
            m = self.masks.setdefault(lab, np.zeros((H, W), np.uint8))
            cv2.fillPoly(m, [np.asarray(o["polygon"], np.int32)], 1)
        z = np.zeros((H, W), np.uint8)
        self.floor = self.masks.get("floor", z)
        self.door = self.masks.get("door", z)
        self.block = np.zeros((H, W), np.uint8)
        for lab in BLOCKING:
            self.block |= self.masks.get(lab, z)
        labelled = np.zeros((H, W), np.uint8)
        for m in self.masks.values():
            labelled |= m
        self.labelled = labelled

        # distance (px) from every pixel to the nearest blocking pixel
        inv = (1 - self.block).astype(np.uint8)
        self.dist_block = cv2.distanceTransform(inv, cv2.DIST_L2, 5) if self.block.any() \
            else np.full((H, W), np.inf, np.float32)

        rows = np.where(self.floor.sum(axis=1) >= 5)[0]
        self.floor_top_y = int(rows.min()) if len(rows) else None

        self.exits, self.target_info = self._find_exits()

    # -- exits -------------------------------------------------------------
    def _door_exits(self):
        out = []
        for o in self.objects:
            if o["label"].lower() != "door":
                continue
            p = np.asarray(o["polygon"], float)
            # threshold line = the part of the door outline that touches the floor.
            # Doors seen at an angle meet the floor along a slanted edge, so the lowest
            # points alone give a tiny threshold.
            contact = self._floor_contact(p)
            if len(contact) >= PARAMS["door_min_contact_pts"]:
                c0 = contact.mean(axis=0)
                axis = np.linalg.svd(contact - c0)[2][0]
                proj = (contact - c0) @ axis
                segment = [(c0 + proj.min() * axis).tolist(), (c0 + proj.max() * axis).tolist()]
                anchor, thr_src = c0, "floor_contact"
            else:
                ymax = p[:, 1].max()
                bottom = p[p[:, 1] >= ymax - PARAMS["door_bottom_band_px"]]
                anchor = np.array([bottom[:, 0].mean(), ymax])
                segment = [[bottom[:, 0].min(), ymax], [bottom[:, 0].max(), ymax]]
                thr_src = "lowest_points"
            x0, y0 = p.min(axis=0)
            x1, y1 = p.max(axis=0)
            contains_target = cv2.pointPolygonTest(
                p.astype(np.float32), tuple(map(float, self.target)), False) >= 0
            out.append({
                "type": "door", "edge": None,
                "anchor_px": anchor, "segment_px": segment, "centroid_px": p.mean(axis=0),
                "bbox_px": [x0, y0, x1, y1], "width_px": x1 - x0,
                "area_frac": poly_area(o["polygon"]) / (self.W * self.H),
                "span_px": None, "status": "detected", "threshold_src": thr_src,
                "threshold_len_px": float(np.hypot(*np.subtract(segment[1], segment[0]))),
                "seg_source": o.get("source"), "confidence": o.get("confidence"),
                "_poly": p.astype(np.int32), "_is_target": contains_target,
            })
        return out

    def _floor_contact(self, poly):
        """Points along a polygon outline (every ~2 px) that lie within a few px of floor."""
        r = PARAMS["door_contact_px"]
        near = cv2.dilate(self.floor, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        pts = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            k = max(int(np.hypot(*(b - a)) / 2), 1)
            for s in np.linspace(0, 1, k, endpoint=False):
                q = a + s * (b - a)
                x, y = int(round(q[0])), int(round(q[1]))
                if 0 <= x < self.W and 0 <= y < self.H and near[y, x]:
                    pts.append(q)
        return np.asarray(pts).reshape(-1, 2)

    def _edge_profile(self, edge):
        """Return (blocked[1D], floor_near[1D], inward_open[1D]) along an edge."""
        b, d, f = PARAMS["edge_band_px"], PARAMS["inward_depth_px"], PARAMS["floor_touch_px"]
        hard = self.block | self.door
        if edge == "top":
            sl = lambda m, k: m[:k, :].T
        elif edge == "bottom":
            sl = lambda m, k: m[-k:, :].T
        elif edge == "left":
            sl = lambda m, k: m[:, :k]
        else:
            sl = lambda m, k: m[:, -k:]
        blocked = sl(hard, b).any(axis=1)
        floor_near = sl(self.floor, f).any(axis=1)
        inward_open = 1.0 - sl(hard, d).mean(axis=1)
        return blocked, floor_near, inward_open

    def _edge_exits(self):
        out, target_gap = [], None
        for edge in ("top", "bottom", "left", "right"):
            blocked, floor_near, inward_open = self._edge_profile(edge)
            n = len(blocked)
            i = 0
            while i < n:
                if blocked[i]:
                    i += 1
                    continue
                j = i
                while j < n and not blocked[j]:
                    j += 1
                a, bnd = i, j - 1            # inclusive span
                i = j
                if bnd - a + 1 < PARAMS["min_gap_px"]:
                    continue
                if inward_open[a:bnd + 1].mean() < PARAMS["inward_open_frac"]:
                    continue
                mid = (a + bnd) / 2
                if edge in ("top", "bottom"):
                    yy = 0 if edge == "top" else self.H - 1
                    anchor = np.array([mid, yy])
                    segment = [[a, yy], [bnd, yy]]
                else:
                    xx = 0 if edge == "left" else self.W - 1
                    anchor = np.array([xx, mid])
                    segment = [[xx, a], [xx, bnd]]

                # unlabelled != walkable (dark walls are often unlabelled), so an edge
                # gap only counts when the floor actually reaches it
                if not floor_near[a:bnd + 1].any():
                    continue
                status = "detected"

                ex = {
                    "type": "edge", "edge": edge, "anchor_px": anchor, "segment_px": segment,
                    "centroid_px": anchor, "bbox_px": None,
                    "width_px": bnd - a + 1, "area_frac": None,
                    "span_px": [a, bnd], "status": status,
                    "seg_source": "floor",
                    "confidence": None, "_poly": None, "_is_target": False,
                }
                # the gap the target door sits in is the target, not a competitor
                tx, ty = self.target
                on_this_edge = {"top": ty <= 10, "bottom": ty >= self.H - 11,
                                "left": tx <= 10, "right": tx >= self.W - 11}[edge]
                along = tx if edge in ("top", "bottom") else ty
                if on_this_edge and a <= along <= bnd:
                    ex["_is_target"] = True
                    target_gap = ex
                out.append(ex)
        return out, target_gap

    def _floor_tip_exits(self):
        """Exits where the floor sticks out and is then cut off by a wall
        (a corridor continuing behind an occluding wall).

        1. Skeleton endpoints of the floor mask are candidate tips.
        2. Keep tips that protrude beyond the floor 'body' (floor opened with a disk).
        3. Look beyond the tip, in its direction: the nearest blocker decides.
           wall/window -> exit; door -> it is that door's floor (skip); nothing -> noise.
        """
        from skimage.morphology import skeletonize
        P = PARAMS
        if not self.floor.any():
            return []
        fl = cv2.morphologyEx(self.floor, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # keep only floor patches big enough to walk on
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(fl, connectivity=8)
        if n_cc <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.where(areas >= P["tip_min_floor_frac"] * areas.max())[0]
        fl = np.isin(cc, keep).astype(np.uint8)
        sk = skeletonize(fl.astype(bool))
        pts = set(zip(*np.nonzero(sk)))
        nb = lambda q: [(q[0] + dy, q[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dy or dx) and (q[0] + dy, q[1] + dx) in pts]
        r = P["tip_open_radius_px"]
        body = cv2.morphologyEx(fl, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        d_body = cv2.distanceTransform((1 - body).astype(np.uint8), cv2.DIST_L2, 5)
        d_floor = cv2.distanceTransform(fl, cv2.DIST_L2, 5)
        lab = np.zeros((self.H, self.W), np.uint8)          # 1 = wall/window, 2 = door
        lab[self.block > 0] = 1
        lab[self.door > 0] = 2
        m, R = P["edge_margin_px"], P["tip_look_px"]
        cos_min = math.cos(math.radians(P["tip_cone_deg"]))

        cands = []
        for e in [q for q in pts if len(nb(q)) == 1]:
            y, x = e
            if x < m or y < m or x > self.W - 1 - m or y > self.H - 1 - m:
                continue                                    # frame-edge exits handled elsewhere
            protr = float(d_body[y, x])
            if protr < P["tip_min_protrusion_px"]:
                continue
            path, prev, cur = [e], None, e                  # walk back along the branch
            while len(path) <= P["tip_dir_back_px"]:
                nxt = [q for q in nb(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            back = path[-1]
            d = np.array([x - back[1], y - back[0]], float)
            if np.linalg.norm(d) < 1e-6:
                continue
            d /= np.linalg.norm(d)
            # nearest blocker inside the look-ahead cone
            y0, y1 = max(0, y - R), min(self.H, y + R + 1)
            x0, x1 = max(0, x - R), min(self.W, x + R + 1)
            yy, xx = np.nonzero(lab[y0:y1, x0:x1])
            if not len(yy):
                continue
            vx, vy = xx + x0 - x, yy + y0 - y
            dist = np.hypot(vx, vy)
            ok = (dist > 0) & (dist <= R) & ((vx * d[0] + vy * d[1]) >= cos_min * dist)
            if not ok.any():
                continue
            k = np.argmin(np.where(ok, dist, np.inf))
            if lab[yy[k] + y0, xx[k] + x0] != 1:
                continue                                    # a door is beyond: not a new exit
            half = max(float(np.max([d_floor[q] for q in path])), 5.0)
            perp = np.array([-d[1], d[0]])
            tip = np.array([x, y], float)
            cands.append({
                "type": "floor_tip", "edge": None, "anchor_px": tip,
                "segment_px": [(tip - perp * half).tolist(), (tip + perp * half).tolist()],
                "centroid_px": tip, "bbox_px": None, "width_px": 2 * half, "area_frac": None,
                "span_px": None, "status": "inferred", "seg_source": "floor",
                "confidence": None, "direction": d.tolist(), "protrusion_px": protr,
                "blocked_by": "wall", "_poly": None, "_is_target": False,
            })
        cands.sort(key=lambda c: -c["protrusion_px"])
        kept = []
        for c in cands:
            if all(np.hypot(*(c["anchor_px"] - k["anchor_px"])) > P["tip_merge_px"] for k in kept):
                kept.append(c)
        return kept

    def _find_exits(self):
        doors = sorted(self._door_exits(), key=lambda e: e["anchor_px"][0])
        edges, target_gap = self._edge_exits()
        edges += self._floor_tip_exits()
        target_door = next((d for d in doors if d["_is_target"]), None)
        exits = [e for e in doors + edges if not e["_is_target"]]
        if self.manual_exits:
            # hand-annotated exits for this camera replace the automatic ones
            exits = []
            for m in self.manual_exits:
                seg = np.asarray(m["segment_px"], float)
                exits.append({
                    "type": m.get("type", "manual"), "edge": m.get("edge"),
                    "anchor_px": seg.mean(axis=0), "segment_px": seg.tolist(),
                    "centroid_px": seg.mean(axis=0), "bbox_px": None,
                    "width_px": float(np.hypot(*(seg[1] - seg[0]))), "area_frac": None,
                    "span_px": None, "status": "manual", "seg_source": None,
                    "confidence": None, "_poly": None, "_is_target": False,
                })
        # target as a segment: visible door threshold > edge gap it sits in > the point itself
        if target_door is not None:
            tseg = target_door["segment_px"]
        elif target_gap is not None:
            tseg = target_gap["segment_px"]
        else:
            tseg = [self.target.tolist(), self.target.tolist()]
        target_info = {
            "center_px": self.target,
            "door_polygon_visible": target_door is not None,
            "edge_gap": target_gap["span_px"] if target_gap else None,
            "segment_px": tseg,
        }
        return exits, target_info

    # -- queries -----------------------------------------------------------
    def inside(self, mask, pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < self.W and 0 <= y < self.H:
            return bool(mask[y, x])
        return False

    def dist_to_block(self, pt):
        x = int(np.clip(round(pt[0]), 0, self.W - 1))
        y = int(np.clip(round(pt[1]), 0, self.H - 1))
        return float(self.dist_block[y, x])

    def overlap_frac(self, bbox):
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.W), min(y1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        region = (self.block | self.door)[y0:y1, x0:x1]
        return float(region.mean())

    def inside_exit(self, pt, tol=5):
        """Index of the door exit whose polygon (dilated by tol) contains pt."""
        for i, e in enumerate(self.exits):
            if e["_poly"] is None:
                continue
            d = cv2.pointPolygonTest(e["_poly"].astype(np.float32),
                                     (float(pt[0]), float(pt[1])), True)
            if d >= -tol:
                return i
        return None

    def summary(self, background_path=None):
        tot = self.W * self.H
        frac = lambda m: float(m.sum()) / tot
        bright = contrast = None
        if background_path and Path(background_path).is_file():
            g = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                bright, contrast = float(g.mean()) / 255, float(g.std()) / 255
        return {
            "target": {
                "center_px": rnd(self.target_info["center_px"], 1),
                "door_polygon_visible": self.target_info["door_polygon_visible"],
                "edge_gap_px": self.target_info["edge_gap"],
                "segment_px": rnd(self.target_info["segment_px"], 1),
            },
            "exits": [
                {k: rnd(v, 2) if k in ("anchor_px", "segment_px", "centroid_px", "bbox_px", "width_px",
                                       "area_frac", "direction", "protrusion_px", "threshold_len_px")
                 else v for k, v in e.items() if not k.startswith("_")}
                for e in self.exits
            ],
            "counts": {
                "n_exits": len(self.exits),
                "n_door": sum(e["type"] == "door" for e in self.exits),
                "n_edge": sum(e["type"] == "edge" for e in self.exits),
                "n_floor_tip": sum(e["type"] == "floor_tip" for e in self.exits),
                "n_manual": sum(e["status"] == "manual" for e in self.exits),
            },
            "areas": {
                "floor_frac": rnd(frac(self.floor), 4),
                "wall_frac": rnd(frac(self.masks.get("wall", np.zeros(1))), 4),
                "window_frac": rnd(frac(self.masks.get("window", np.zeros(1))), 4),
                "door_frac": rnd(frac(self.door), 4),
                "unlabelled_frac": rnd(1 - frac(self.labelled), 4),
            },
            "floor_top_y_px": self.floor_top_y,
            "brightness_mean": rnd(bright, 4),
            "brightness_contrast": rnd(contrast, 4),
            "n_polygons_used": len(self.objects),
            "n_polygons_dropped": self.dropped,
        }


# ---------------------------------------------------------------- keypoints
def load_keypoints(traj, pose, W, H):
    """Per frame: (17,3) array or None. Trajectory keypoints first, pose sidecar as fallback."""
    pose_by_idx = {}
    if pose:
        for fr in pose.get("frames", []):
            if fr.get("detected") and fr.get("landmarks"):
                arr = np.full((17, 3), np.nan)
                arr[:, 2] = 0.0
                for name, v in fr["landmarks"].items():
                    if name in K:
                        arr[K[name]] = [v["x"] * W, v["y"] * H, v.get("visibility", 0.0)]
                pose_by_idx[fr["frame_index"]] = arr
    out = []
    for fr in traj["frames"]:
        kp = fr.get("keypoints")
        if kp is not None:
            out.append(np.asarray(kp, float))
        else:
            out.append(pose_by_idx.get(fr["frame_index"]))
    return out


def ground_point(kp, bbox, H):
    """Mid-ankle if both ankles are visible, else bbox bottom-centre."""
    c = PARAMS["kp_conf_min"]
    la, ra = kp[K["left_ankle"]], kp[K["right_ankle"]]
    if la[2] >= c and ra[2] >= c:
        return (la[:2] + ra[:2]) / 2, "ankles"
    if bbox is not None:
        return np.array([(bbox[0] + bbox[2]) / 2, bbox[3]]), "bbox_bottom"
    return None, None


def mid(kp, a, b):
    c = PARAMS["kp_conf_min"]
    if kp[K[a]][2] >= c and kp[K[b]][2] >= c:
        return (kp[K[a]][:2] + kp[K[b]][:2]) / 2
    return None


def pose_features(kp, bh, heading_deg):
    c = PARAMS["kp_conf_min"]
    vis = lambda n: kp[K[n]][2] >= c
    P = lambda n: kp[K[n]][:2]
    f = {}
    conf = kp[:, 2]
    f["kp_conf_mean"] = float(np.nanmean(conf))
    f["n_kp_visible"] = int((conf >= c).sum())

    nose_ankle = None
    am = mid(kp, "left_ankle", "right_ankle")
    if vis("nose") and am is not None:
        nose_ankle = am[1] - P("nose")[1]
    f["body_height_px"] = nose_ankle

    sw = None
    if vis("left_shoulder") and vis("right_shoulder"):
        sv = P("left_shoulder") - P("right_shoulder")
        sw = float(np.hypot(*sv))
        f["shoulder_width_bh"] = sw / bh if bh else None
        sh_ang = math.degrees(math.atan2(sv[1], sv[0]))
        f["shoulder_angle_deg"] = sh_ang
        # 0 = shoulders perpendicular to walking direction (normal gait), 90 = walking sideways
        if heading_deg is not None:
            d = ang_diff(sh_ang, heading_deg + 90) % 180
            f["torso_vs_motion_deg"] = min(d, 180 - d)
        else:
            f["torso_vs_motion_deg"] = None
    else:
        f["shoulder_width_bh"] = f["shoulder_angle_deg"] = f["torso_vs_motion_deg"] = None

    hm = mid(kp, "left_hip", "right_hip")
    f["hip_width_bh"] = (float(np.hypot(*(P("left_hip") - P("right_hip")))) / bh
                         if hm is not None and bh else None)

    # head yaw proxy: nose offset from ear midpoint, relative to shoulder width
    em = mid(kp, "left_ear", "right_ear")
    if vis("nose") and sw:
        ref = em if em is not None else (P("left_ear") if vis("left_ear") else
                                         P("right_ear") if vis("right_ear") else None)
        f["head_yaw_proxy"] = float((P("nose")[0] - ref[0]) / sw) if ref is not None else None
    else:
        f["head_yaw_proxy"] = None
    f["ear_vis_asym"] = float(kp[K["left_ear"]][2] - kp[K["right_ear"]][2])

    if vis("left_ankle") and vis("right_ankle") and bh:
        f["ankle_sep_bh"] = float(np.hypot(*(P("left_ankle") - P("right_ankle")))) / bh
    else:
        f["ankle_sep_bh"] = None

    if hm is not None and am is not None and bh:
        f["leg_length_bh"] = float(np.hypot(*(am - hm))) / bh
    else:
        f["leg_length_bh"] = None

    if vis("left_wrist") and vis("right_wrist") and hm is not None and bh:
        f["wrist_spread_bh"] = float(np.hypot(*(P("left_wrist") - P("right_wrist")))) / bh
    else:
        f["wrist_spread_bh"] = None
    return f


# ---------------------------------------------------------------- main extraction
def extract(traj, scene_json, pose=None, background=None, sources=None, manual_exits=None):
    W, H = int(traj["frame_width"]), int(traj["frame_height"])
    target = traj.get("door_center", [W / 2, H - 1])
    scene = Scene(scene_json, W, H, target, manual_exits)
    kps = load_keypoints(traj, pose, W, H)
    frames_in = traj["frames"]
    n = len(frames_in)

    # ---- pass 1: raw per-frame geometry
    t = np.array([fr["t"] for fr in frames_in], float)
    det = np.array([kps[i] is not None for i in range(n)])
    gp = np.full((n, 2), np.nan)
    gsrc = [None] * n
    bbox_h = np.full(n, np.nan)
    trunc = np.zeros(n, bool)
    for i, fr in enumerate(frames_in):
        if not det[i]:
            continue
        bb = fr.get("bbox")
        p, src = ground_point(kps[i], bb, H)
        if p is not None:
            gp[i], gsrc[i] = p, src
        if bb is not None:
            bbox_h[i] = bb[3] - bb[1]
            trunc[i] = bb[3] >= H - 2 or bb[1] <= 1
    # causal body-height reference: median of last 3 non-truncated bbox heights
    bh_ref = np.full(n, np.nan)
    hist = []
    for i in range(n):
        if det[i] and not trunc[i] and np.isfinite(bbox_h[i]):
            hist.append(bbox_h[i])
        if det[i] and hist:
            bh_ref[i] = float(np.median(hist[-3:]))

    # ---- pass 2: causal smoothing + backward-difference kinematics
    w = PARAMS["smooth_window"]
    sm = np.full((n, 2), np.nan)
    for i in range(n):
        if not np.isfinite(gp[i]).all():
            continue
        lo = i
        while lo > 0 and i - lo + 1 < w and np.isfinite(gp[lo - 1]).all():
            lo -= 1
        sm[i] = gp[lo:i + 1].mean(axis=0)

    vel = np.full((n, 2), np.nan)
    for i in range(1, n):
        if np.isfinite(sm[i]).all() and np.isfinite(sm[i - 1]).all():
            vel[i] = (sm[i] - sm[i - 1]) / max(t[i] - t[i - 1], 1e-6)

    # looming: d ln(bbox height)/dt. Pixel speed hides motion toward/away from the camera
    # (foreshortening); apparent size growth captures it. 1/looming = time-to-contact tau.
    hs = np.full(n, np.nan)
    for i in range(n):
        good = [j for j in range(max(0, i - w + 1), i + 1)
                if det[j] and not trunc[j] and np.isfinite(bbox_h[j])]
        if det[i] and not trunc[i] and good:
            hs[i] = np.mean(bbox_h[good])
    loom = np.full(n, np.nan)
    for i in range(1, n):
        if np.isfinite(hs[i]) and np.isfinite(hs[i - 1]):
            loom[i] = (math.log(hs[i]) - math.log(hs[i - 1])) / max(t[i] - t[i - 1], 1e-6)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    heading = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))       # 0=right, 90=down (image)

    frames_out = []
    prev = {"speed": None, "heading": None, "t": None}
    exit_anchors = [e["anchor_px"] for e in scene.exits]
    exit_segs = [e["segment_px"] for e in scene.exits]
    tgt_seg = scene.target_info["segment_px"]

    for i, fr in enumerate(frames_in):
        base = {"frame_index": fr["frame_index"], "t": rnd(fr["t"], 4), "detected": bool(det[i])}
        if not det[i]:
            frames_out.append(base)
            prev = {"speed": None, "heading": None, "t": None}
            continue
        bh = bh_ref[i] if np.isfinite(bh_ref[i]) else None
        p = sm[i] if np.isfinite(sm[i]).all() else None
        v = vel[i] if np.isfinite(vel[i]).all() else None
        sp = float(speed[i]) if v is not None else None
        moving = sp is not None and bh and sp / bh >= PARAMS["min_speed_bh_s"]
        hd = float(heading[i]) if moving else None

        # kinematics
        accel = turn = curv = None
        if sp is not None and prev["speed"] is not None:
            dt = max(t[i] - prev["t"], 1e-6)
            accel = (sp - prev["speed"]) / dt
            if hd is not None and prev["heading"] is not None:
                dh = (hd - prev["heading"] + 180) % 360 - 180
                turn = dh / dt
                curv = math.radians(turn) / sp if sp > 1e-6 else None
        kin = {
            "speed_px_s": sp,
            "speed_bh_s": sp / bh if sp is not None and bh else None,
            "accel_bh_s2": accel / bh if accel is not None and bh else None,
            "heading_deg": hd,
            "turn_rate_deg_s": turn,
            "curvature_per_px": curv,
            "looming_rate_per_s": loom[i] if np.isfinite(loom[i]) else None,
            "tau_s": (1 / loom[i]) if np.isfinite(loom[i]) and loom[i] > 1e-3 else None,
        }
        prev = {"speed": sp, "heading": hd, "t": t[i]}

        # position
        pos = {
            "ground_px": p,
            "ground_norm": [p[0] / W, p[1] / H] if p is not None else None,
            "ground_src": gsrc[i],
            "bbox_h_px": bbox_h[i],
            "bh_ref_px": bh,
            "bbox_truncated": bool(trunc[i]),
            # ground from a truncated bbox is the frame edge, not the feet
            "reliable": bool(gsrc[i] == "ankles" or not trunc[i]),
            "on_floor": scene.inside(scene.floor, p) if p is not None else None,
            "inside_exit_idx": scene.inside_exit(p) if p is not None else None,
        }

        # pose
        pf = pose_features(kps[i], bh, hd)

        # scene-relative
        def rel(seg, anchor):
            if p is None:
                return {"dist_px": None, "dist_bh": None, "align_deg": None,
                        "align_center_deg": None, "ttr_s": None}
            dist, near, align = seg_relation(p, seg, hd)
            out = {"dist_px": dist, "dist_bh": dist / bh if bh else None,
                   "align_deg": align, "align_center_deg": None, "ttr_s": None}
            # align_deg is 0 whenever the heading hits the opening anywhere, so wide
            # openings (a corridor filling the frame edge) align with almost everything;
            # align_center_deg keeps the angle to the opening's centre to tell them apart
            vc = np.asarray(anchor, float) - p
            if hd is not None and np.hypot(*vc) > 1e-6:
                out["align_center_deg"] = ang_diff(hd, math.degrees(math.atan2(vc[1], vc[0])))
            if hd is not None and dist > 1e-6:
                approach = float(np.dot(v, (near - p) / dist))
                out["ttr_s"] = dist / approach if approach > 1e-6 else None
            return out

        exits_rel = [rel(s, a) for s, a in zip(exit_segs, exit_anchors)]
        ranked = sorted([k for k, r in enumerate(exits_rel) if r["align_deg"] is not None],
                        key=lambda k: (exits_rel[k]["align_deg"], exits_rel[k]["align_center_deg"]))
        comps = []
        for k in ranked[:PARAMS["n_competitors"]]:
            e = scene.exits[k]
            comps.append({"exit_idx": k, "type": e["type"], "edge": e["edge"],
                          "status": e["status"], **exits_rel[k]})
        comps += [None] * (PARAMS["n_competitors"] - len(comps))
        tr = rel(tgt_seg, scene.target)
        best_comp = comps[0]["align_deg"] if comps[0] else None
        most_aligned = None
        if tr["align_deg"] is not None:
            key = lambda r: (r["align_deg"], r["align_center_deg"] if r["align_center_deg"] is not None else 180)
            most_aligned = "target" if comps[0] is None or key(tr) <= key(comps[0]) else "competitor"

        scene_rel = {
            "dist_block_px": scene.dist_to_block(p) if p is not None and scene.block.any() else None,
            "dist_block_bh": (scene.dist_to_block(p) / bh) if p is not None and bh and scene.block.any() else None,
            "occlusion_frac": scene.overlap_frac(fr["bbox"]) if fr.get("bbox") else None,
            "target": tr,
            "competitors": comps,
            "most_aligned": most_aligned,
            "target_minus_best_competitor_align_deg":
                (tr["align_deg"] - best_comp) if tr["align_deg"] is not None and best_comp is not None else None,
            "exits": exits_rel,
        }
        frames_out.append({**base, "pos": pos, "kin": kin, "pose": pf, "scene_rel": scene_rel})

    # ---- track summary
    valid = np.where(np.isfinite(gp).all(axis=1))[0]
    first, last = (valid[0], valid[-1]) if len(valid) else (None, None)
    origin = None
    if first is not None:
        p0 = gp[first]
        bh0 = bh_ref[first] if np.isfinite(bh_ref[first]) else None
        idx = scene.inside_exit(p0)
        if idx is None and exit_anchors and bh0:
            d = [np.hypot(*(a - p0)) for a in exit_anchors]
            k = int(np.argmin(d))
            if d[k] / bh0 <= PARAMS["origin_max_dist_bh"]:
                idx = k
        if idx is not None:
            e = scene.exits[idx]
            origin = {"exit_idx": idx, "type": e["type"], "edge": e["edge"],
                      "status": e["status"], "anchor_px": rnd(e["anchor_px"], 1),
                      "dist_px": rnd(np.hypot(*(e["anchor_px"] - p0)), 1)}
        else:
            ed = {"left": p0[0], "right": W - p0[0], "top": p0[1], "bottom": H - p0[1]}
            edge = min(ed, key=ed.get)
            origin = {"exit_idx": None, "type": "frame_edge" if bh0 and ed[edge] / bh0 <= 1 else "appeared_mid_frame",
                      "edge": edge, "status": None, "anchor_px": None, "dist_px": rnd(ed[edge], 1)}

    path_len = float(np.nansum(np.hypot(*np.diff(gp[valid], axis=0).T))) if len(valid) > 1 else None
    disp = float(np.hypot(*(gp[last] - gp[first]))) if len(valid) > 1 else None
    med_bh = float(np.nanmedian(bh_ref)) if np.isfinite(bh_ref).any() else None
    spd_bh = speed / bh_ref
    meta = traj.get("meta", {})

    # camera geometry from this track: bbox height grows linearly with ground y under a
    # flat floor; where the fit hits 0 is the horizon (vanishing line) -> tilt proxy
    ok = det & ~trunc & np.isfinite(bbox_h) & np.isfinite(gp[:, 1])
    cam = {"bh_vs_y_slope": None, "bh_vs_y_intercept": None, "bh_vs_y_r2": None,
           "horizon_y_px": None, "n_points": int(ok.sum()), "source": "this_track_only"}
    if ok.sum() >= 5:
        y, h = gp[ok, 1], bbox_h[ok]
        a, b = np.polyfit(y, h, 1)
        pred = a * y + b
        r2 = 1 - ((h - pred) ** 2).sum() / max(((h - h.mean()) ** 2).sum(), 1e-9)
        cam.update(bh_vs_y_slope=a, bh_vs_y_intercept=b, bh_vs_y_r2=r2,
                   horizon_y_px=(-b / a) if abs(a) > 1e-6 else None)

    track_summary = {
        "causal": {
            "origin_exit": origin,
            "first_ground_norm": rnd([gp[first][0] / W, gp[first][1] / H], 4) if first is not None else None,
            "first_bh_px": rnd(bh_ref[first], 1) if first is not None else None,
            "n_concurrent_tracks": None,   # needs multi-track tracker output
        },
        "hindsight": {
            "entry_edge": meta.get("entry_edge"),
            "exit_edge": meta.get("exit_edge"),
            "duration_s": rnd(t[last] - t[first], 3) if first is not None else None,
            "n_frames": n,
            "n_detected": int(det.sum()),
            "detection_rate": rnd(det.mean(), 4),
            "path_length_bh": rnd(path_len / med_bh, 3) if path_len and med_bh else None,
            "displacement_bh": rnd(disp / med_bh, 3) if disp and med_bh else None,
            "path_straightness": rnd(disp / path_len, 4) if disp and path_len else None,
            "mean_speed_bh_s": rnd(np.nanmean(spd_bh), 3) if np.isfinite(spd_bh).any() else None,
            "max_speed_bh_s": rnd(np.nanmax(spd_bh), 3) if np.isfinite(spd_bh).any() else None,
            "last_ground_norm": rnd([gp[last][0] / W, gp[last][1] / H], 4) if last is not None else None,
            "camera_geom": {k: rnd(v, 4) if isinstance(v, float) else v for k, v in cam.items()},
        },
    }

    # ---- assemble
    for fr in frames_out:
        for blk in ("pos", "kin", "pose"):
            if blk in fr:
                fr[blk] = {k: (v if isinstance(v, (str, bool)) or v is None else rnd(v, 3))
                           for k, v in fr[blk].items()}
        if "scene_rel" in fr:
            fr["scene_rel"] = json.loads(json.dumps(fr["scene_rel"], default=lambda o: rnd(o, 3)),
                                         parse_float=lambda s: round(float(s), 3))

    info = {
        "schema_version": SCHEMA_VERSION,
        "clip": meta.get("source_video") or (sources or {}).get("trajectory"),
        "label": traj.get("label"),
        "sources": sources or {},
        "segmentation": {
            "labels_present": sorted(scene.masks.keys()),
            "sources_present": sorted({str(o.get("source")) for o in scene.objects}),
        },
        "depth": None,
        "trajectory_meta": {k: meta.get(k) for k in
                            ("session_id", "git_commit", "effective_fps", "record_fps", "distort", "tracker")},
        "frame_size": [W, H],
        "params": PARAMS,
        "units": {"_px": "pixels", "_bh": "body heights (causal median of last 3 untruncated bbox heights)",
                  "_s": "seconds", "_deg": "degrees (image coords: 0=right, 90=down)",
                  "_norm": "fraction of frame width/height"},
        "notes": "data.frames.* are causal (current + past frames only). "
                 "data.track_summary.hindsight uses the whole track - do not use for online prediction.",
    }
    data = {"scene_summary": scene.summary(background), "track_summary": track_summary,
            "frames": frames_out}
    return {"info": info, "data": data}, scene


# ---------------------------------------------------------------- catalog
def C(name, level, group, unit, needs, causal, desc):
    return {"name": name, "level": level, "group": group, "unit": unit,
            "needs": needs, "causal": causal, "description": desc}

T, P_, S = "trajectory", "pose", "scene"
CATALOG = [
    C("pos.ground_px", "frame", "pos", "px", [T, P_], True, "Smoothed ground contact point (mid-ankle, else bbox bottom)"),
    C("pos.ground_norm", "frame", "pos", "norm", [T, P_], True, "Ground point / frame size"),
    C("pos.bh_ref_px", "frame", "pos", "px", [T], True, "Body-height scale used for all _bh units"),
    C("pos.bbox_truncated", "frame", "pos", "bool", [T], True, "Bbox touches top/bottom frame edge"),
    C("pos.on_floor", "frame", "pos", "bool", [T, S], True, "Ground point inside floor mask"),
    C("pos.inside_exit_idx", "frame", "pos", "index", [T, S], True, "Door exit whose polygon contains the ground point"),
    C("kin.speed_bh_s", "frame", "kin", "bh/s", [T], True, "Ground speed in body heights per second"),
    C("kin.accel_bh_s2", "frame", "kin", "bh/s^2", [T], True, "Change of speed"),
    C("kin.heading_deg", "frame", "kin", "deg", [T], True, "Walking direction in image"),
    C("kin.turn_rate_deg_s", "frame", "kin", "deg/s", [T], True, "Heading change rate"),
    C("kin.curvature_per_px", "frame", "kin", "rad/px", [T], True, "Path curvature"),
    C("kin.looming_rate_per_s", "frame", "kin", "1/s", [T], True, "d ln(bbox height)/dt; >0 = approaching camera (target door)"),
    C("kin.tau_s", "frame", "kin", "s", [T], True, "Time-to-contact = 1/looming rate (Lee 1976)"),
    C("pos.reliable", "frame", "pos", "bool", [T, P_], True, "False when ground point comes from a truncated bbox"),
    C("pose.kp_conf_mean", "frame", "pose", "0-1", [P_], True, "Mean keypoint confidence (quality / lighting proxy)"),
    C("pose.n_kp_visible", "frame", "pose", "count", [P_], True, "Keypoints above confidence threshold"),
    C("pose.body_height_px", "frame", "pose", "px", [P_], True, "Nose to mid-ankle height"),
    C("pose.shoulder_width_bh", "frame", "pose", "bh", [P_], True, "Shoulder width; drops when body turns sideways"),
    C("pose.hip_width_bh", "frame", "pose", "bh", [P_], True, "Hip width"),
    C("pose.shoulder_angle_deg", "frame", "pose", "deg", [P_], True, "Shoulder line orientation"),
    C("pose.torso_vs_motion_deg", "frame", "pose", "deg", [P_, T], True, "0 = torso square to walking direction, 90 = sideways"),
    C("pose.head_yaw_proxy", "frame", "pose", "shoulder widths", [P_], True, "Nose offset from ear midpoint"),
    C("pose.ear_vis_asym", "frame", "pose", "-1..1", [P_], True, "Left minus right ear confidence (head turn cue)"),
    C("pose.ankle_sep_bh", "frame", "pose", "bh", [P_], True, "Ankle separation (gait phase / stride)"),
    C("pose.leg_length_bh", "frame", "pose", "bh", [P_], True, "Hip to ankle length"),
    C("pose.wrist_spread_bh", "frame", "pose", "bh", [P_], True, "Distance between wrists (arm swing / carrying)"),
    C("scene_rel.dist_block_bh", "frame", "scene_rel", "bh", [T, S], True, "Distance to nearest wall/window"),
    C("scene_rel.occlusion_frac", "frame", "scene_rel", "0-1", [T, S], True, "Share of bbox overlapping wall/window/door"),
    C("scene_rel.target.dist_bh", "frame", "scene_rel", "bh", [T], True, "Distance to target door"),
    C("scene_rel.target.align_deg", "frame", "scene_rel", "deg", [T], True, "Angle between heading and direction to target"),
    C("scene_rel.target.align_center_deg", "frame", "scene_rel", "deg", [T], True, "Angle between heading and direction to the target's centre"),
    C("scene_rel.target.ttr_s", "frame", "scene_rel", "s", [T], True, "Time to reach target at current approach speed"),
    C("scene_rel.competitors[k].*", "frame", "scene_rel", "mixed", [T, S], True, "Top-k most aligned non-target exits (null-padded)"),
    C("scene_rel.most_aligned", "frame", "scene_rel", "category", [T, S], True, "'target' or 'competitor'"),
    C("scene_rel.target_minus_best_competitor_align_deg", "frame", "scene_rel", "deg", [T, S], True, "<0 means target explains motion better"),
    C("scene_rel.exits[i].*", "frame", "scene_rel", "mixed", [T, S], True, "Per-exit distance/alignment (variable length)"),
    C("scene_summary.counts.*", "scene", "scene", "count", [S], True, "Number of exits by type/status"),
    C("scene_summary.areas.*", "scene", "scene", "fraction", [S], True, "Floor/wall/window/door/unlabelled area shares"),
    C("scene_summary.floor_top_y_px", "scene", "scene", "px", [S], True, "Highest floor row (horizon proxy)"),
    C("scene_summary.brightness_mean", "scene", "scene", "0-1", ["background"], True, "Background brightness (lighting)"),
    C("track_summary.causal.origin_exit", "track", "track", "object", [T, S], True, "Exit the person came from"),
    C("track_summary.causal.first_bh_px", "track", "track", "px", [T], True, "Body height at first detection"),
    C("track_summary.hindsight.*", "track", "track", "mixed", [T], False, "Whole-track stats: NOT for online prediction"),
    C("track_summary.hindsight.camera_geom.*", "track", "camera", "mixed", [T], False, "bbox height vs y fit, horizon estimate"),
]


# ---------------------------------------------------------------- debug overlay
def draw_debug(scene, result, out_path, background=None):
    W, H = scene.W, scene.H
    img = None
    if background and Path(background).is_file():
        img = cv2.imread(str(background))
    if img is None:
        img = np.full((H, W, 3), 40, np.uint8)
    over = img.copy()
    over[scene.floor > 0] = (60, 160, 60)
    over[scene.block > 0] = (160, 90, 60)
    over[scene.door > 0] = (40, 140, 220)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for k, e in enumerate(scene.exits):
        col = (0, 255, 255) if e["status"] == "detected" else (0, 140, 255)
        if e["type"] == "edge":
            a, b = e["span_px"]
            p1, p2 = {"top": ((a, 1), (b, 1)), "bottom": ((a, H - 2), (b, H - 2)),
                      "left": ((1, a), (1, b)), "right": ((W - 2, a), (W - 2, b))}[e["edge"]]
            cv2.line(img, p1, p2, col, 5)
        x, y = map(int, e["anchor_px"])
        cv2.circle(img, (x, y), 5, col, -1)
        cv2.putText(img, f"{k}:{e['type'][0]}/{e['status'][0]}", (min(x + 6, W - 60), max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    s0, s1 = scene.target_info["segment_px"]
    cv2.line(img, tuple(map(int, s0)), (int(s1[0]), int(s1[1])), (0, 0, 255), 3)
    for e in scene.exits:
        if e["type"] != "edge":
            a, b = e["segment_px"]
            col = (0, 255, 255) if e["status"] == "detected" else (0, 140, 255)
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), col, 3)
            if e.get("direction") is not None:          # arrow: where the floor continues
                tip = np.asarray(e["anchor_px"], float)
                cv2.arrowedLine(img, tuple(map(int, tip)),
                                tuple(map(int, tip + 25 * np.asarray(e["direction"]))), col, 2, tipLength=0.4)
    tx, ty = map(int, scene.target)
    cv2.drawMarker(img, (tx, min(ty, H - 4)), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
    pts = [f["pos"]["ground_px"] for f in result["data"]["frames"] if f.get("pos") and f["pos"]["ground_px"]]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 2)
    if pts:
        cv2.circle(img, tuple(map(int, pts[0])), 6, (0, 255, 0), -1)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------- IO
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_clip(traj_p, scene_p, pose_p=None, bg_p=None, out_p=None, debug=False, exits_p=None):
    traj_p = Path(traj_p)
    pose = load(pose_p) if pose_p and Path(pose_p).is_file() else None
    sources = {"trajectory": traj_p.name, "scene": Path(scene_p).name,
               "pose": Path(pose_p).name if pose else None,
               "background": Path(bg_p).name if bg_p and Path(bg_p).is_file() else None,
               "depth": None}
    manual = load(exits_p)["exits"] if exits_p and Path(exits_p).is_file() else None
    sources["manual_exits"] = Path(exits_p).name if manual else None
    result, scene = extract(load(traj_p), load(scene_p), pose, bg_p, sources, manual)
    out_p = Path(out_p) if out_p else traj_p.with_name(f"{traj_p.stem}_features.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if debug:
        draw_debug(scene, result, out_p.with_name(out_p.stem + "_debug.png"), bg_p)
    return out_p


def extract_features(trajectory_path: str, debug=False, exits_p: str = None):
    if not trajectory_path:
        raise SystemExit("[Extract Features] : Please specify trajectory file path")
    
    traj_path = Path(trajectory_path)
    back_path_json =  traj_path.with_name(f"{traj_path.stem}_background.json")
    back_path_png = traj_path.with_name(f"{traj_path.stem}_background.png")
    pose_path_json =  traj_path.with_name(f"{traj_path.stem}_pose.json")
    out_path_json =  traj_path.with_name(f"{traj_path.stem}_features.json")
    
    out_path = run_clip(
            traj_path, 
            back_path_json, 
            pose_path_json,
            back_path_png, 
            out_path_json, 
            debug, 
            exits_p
        )
    cat_dir = Path(out_path).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj")
    ap.add_argument("--scene", help="default: <stem>_background.json")
    ap.add_argument("--pose")
    ap.add_argument("--background")
    ap.add_argument("--out")
    ap.add_argument("--exits", help='optional hand-annotated exits: {"exits": [{"type": ..., "segment_px": [[x,y],[x,y]]}]}')
    ap.add_argument("--input-dir", help="batch: finds <stem>.json + scene_<stem>.json pairs")
    ap.add_argument("--out-dir")
    ap.add_argument("--debug", action="store_true", help="also write an overlay PNG with exits and trajectory")
    a = ap.parse_args()

    outs = []
    if a.input_dir:
        d = Path(a.input_dir)
        od = Path(a.out_dir) if a.out_dir else d
        for traj_p in sorted(d.glob("*.json")):
            s = traj_p.stem
            if s.startswith(("features_", "exits_")) or s == "feature_catalog" \
                    or s.endswith(("_pose", "_background")):
                continue
            scene_p = d / f"{s}_background.json"
            if not scene_p.is_file():
                print(f"skip {s}: no {scene_p.name}")
                continue
            outs.append(run_clip(traj_p, scene_p, d / f"{s}_pose.json", d / f"{s}_background.png",
                                 od / f"features_{s}.json", a.debug,
                                 a.exits or d / f"exits_{s}.json"))
        cat_dir = od
    else:
        if not a.traj:
            ap.error("give --traj, or --input-dir")
        tp = Path(a.traj)
        side = lambda suffix, given: given or tp.with_name(f"{tp.stem}{suffix}")
        outs.append(run_clip(tp, side("_background.json", a.scene), side("_pose.json", a.pose),
                             side("_background.png", a.background), a.out, a.debug, a.exits))
        cat_dir = Path(outs[0]).parent
    with open(cat_dir / "feature_catalog.json", "w", encoding="utf-8") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "features": CATALOG}, f, indent=1)
    for o in outs:
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
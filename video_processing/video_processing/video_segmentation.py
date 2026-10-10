"""Segment the static scene of videos and export each as JSON.

Flow: video -> median background image (background_extraction.py) ->
SAM 3 masks -> hand annotation in the FiftyOne App -> <dataset name>.json
with pixel-coordinate polygons and <dataset name>_overlay.png showing those
polygons on the background.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

from background_extraction import median_background_over_a_video

# Datasets that are already segmented: only opened, annotated and exported
EXISTING_DATASETS = [] # <- everything filled here will be able to be changed (hand annotation) e.g. "20250916_095917_id7_enter", "20250916_095917_id93_enter"

# Videos still to segment: each becomes a dataset named scene_<video name>
VIDEOS = ["20251010_132037_id181_enter.mp4"]  # e.g. ["20250916_095917_id7_enter.mp4"]

PROMPTS = ["floor", "wall", "door", "window"] # <- ADD OR REMOVE ITEMS TO SEGMENT (each item is another pass through the image)
THRESHOLD = 0.7  # minimum confidence for a SAM 3 object to be kept

# Surfaces: overlapping or touching pieces with these labels become one object.s
# Everything else (doors, chairs, ...) stays a separate object per instance.
MERGE_LABELS = {"floor", "wall"}
MIN_AREA = 100   # pixels; leftovers smaller than this are dropped
TOLERANCE = 1.0  # pixels; how much a polygon outline may be simplified


def load_model():
    foz.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")
    model = foz.load_zoo_model("facebook/sam3")
    model.operation = "concept_segmentation"
    model.prompt = PROMPTS
    model.threshold = THRESHOLD
    return model
 
 
def segment_image(image, name, model):
    """Runs SAM 3 once and stores the result in a persistent dataset.
 
    The name is set when the dataset is created and never changed after,
    so the App URL stays valid.
    """
    dataset = fo.Dataset.from_images([image], name=name, persistent=True)
    dataset.apply_model(model, label_field="scene")
    return dataset
 
 
# ---------------------------------------------------------------- masks
 
def detection_mask(det, w, h):
    """Full-image mask of a detection: its instance mask, or its box if it has none."""
    x, y, bw, bh = det.bounding_box
    x0, y0 = max(0, round(x * w)), max(0, round(y * h))
    x1, y1 = min(w, round((x + bw) * w)), min(h, round((y + bh) * h))
 
    full = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return full
 
    mask = det.get_mask() if hasattr(det, "get_mask") else det.mask
    if mask is None:  # hand-drawn box
        full[y0:y1, x0:x1] = True
    else:  # the instance mask covers the box, so stretch it to the box size
        patch = cv2.resize(np.asarray(mask).astype(np.uint8), (x1 - x0, y1 - y0),
                           interpolation=cv2.INTER_NEAREST)
        full[y0:y1, x0:x1] = patch > 0
    return full
 
 
def polyline_mask(polyline, w, h):
    """Full-image mask of a hand-drawn polygon."""
    shapes = [np.array([[round(x * w), round(y * h)] for x, y in shape], np.int32)
              for shape in polyline.points]
    shapes = [s for s in shapes if len(s) >= 3]
    full = np.zeros((h, w), np.uint8)
    if shapes:
        cv2.fillPoly(full, shapes, 1)
    return full.astype(bool)
 
 
def collect_objects(sample, w, h):
    """Reads every label field. Returns (kept objects, dropped SAM objects)."""
    kept, dropped = [], []
    for field in sample.field_names:
        value = sample[field]
 
        if isinstance(value, fo.Detections):
            for det in value.detections:
                label = det.label.lower()
                # Hand annotations have no confidence and are always kept
                if det.confidence is not None and det.confidence < THRESHOLD:
                    dropped.append((label, det.confidence))
                    continue
                kept.append({"label": label, "source": field,
                             "confidence": det.confidence,
                             "mask": detection_mask(det, w, h)})
 
        elif isinstance(value, fo.Polylines):
            for pl in value.polylines:
                kept.append({"label": pl.label.lower(), "source": field,
                             "confidence": None,
                             "mask": polyline_mask(pl, w, h)})
 
    return kept, dropped
 
 
# ------------------------------------------------------------- cleaning
 
def resolve_overlaps(raw, w, h):
    """Turns overlapping masks into polygons where each pixel has one owner.
 
    Where two objects overlap, the winner is:
      1. a hand annotation over a SAM 3 object (you drew it on purpose);
      2. otherwise the smaller object (a door sits inside a wall, a chair
         stands on the floor, so the smaller one is the more specific).
    Pieces with a label in MERGE_LABELS are fused into one object first.
    """
    def rank(o):  # lower sorts first and wins
        return (o["confidence"] is not None, int(o["mask"].sum()))
 
    ranked = sorted(raw, key=rank)
 
    groups, surface, group_of = [], {}, []
    for o in ranked:
        if o["label"] in MERGE_LABELS:
            if o["label"] not in surface:
                surface[o["label"]] = len(groups)
                groups.append({"label": o["label"], "pieces": []})
            g = surface[o["label"]]
        else:
            g = len(groups)
            groups.append({"label": o["label"], "pieces": []})
        groups[g]["pieces"].append(o)
        group_of.append(g)
 
    # Paint from the weakest to the strongest, so the strongest ends up on top
    owner = np.zeros((h, w), np.int32)
    for o, g in reversed(list(zip(ranked, group_of))):
        owner[o["mask"]] = g + 1
 
    def simplify(contour):
        return cv2.approxPolyDP(contour, TOLERANCE, True).reshape(-1, 2).tolist()
 
    objects = []
    for g, group in enumerate(groups):
        mask = (owner == g + 1).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue  # completely covered by stronger objects
 
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:  # this contour is a hole, handled below
                continue
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            polygon = simplify(contour)
            if len(polygon) < 3:
                continue
 
            holes, j = [], hierarchy[0][i][2]
            while j != -1:
                if cv2.contourArea(contours[j]) >= MIN_AREA:
                    hole = simplify(contours[j])
                    if len(hole) >= 3:
                        holes.append(hole)
                j = hierarchy[0][j][0]
 
            # Source and confidence come from the pieces that make up this part
            part = np.zeros((h, w), np.uint8)
            cv2.drawContours(part, contours, i, 1, -1)
            pieces = [p for p in group["pieces"] if (p["mask"] & (part > 0)).any()]
            scores = [p["confidence"] for p in pieces if p["confidence"] is not None]
 
            objects.append({
                "label": group["label"],
                "source": "+".join(sorted({p["source"] for p in pieces})),
                "confidence": max(scores) if scores else None,
                "polygon": polygon,
                "holes": holes,
            })
 
    return objects
 
 
def object_mask(obj, w, h):
    """Rasterises an exported object: its polygon minus its holes."""
    full = np.zeros((h, w), np.uint8)
    shapes = [np.array(obj["polygon"], np.int32)]
    shapes += [np.array(hole, np.int32) for hole in obj["holes"]]
    cv2.fillPoly(full, shapes, 1)  # even-odd rule leaves the holes empty
    return full
 
 
# --------------------------------------------------------------- export
 
def export_scene(dataset, out):
    """Writes the cleaned scene to JSON and a matching overlay picture."""
    dataset.reload()  # pick up annotations made in the App
    sample = dataset.first()
    h, w = cv2.imread(sample.filepath).shape[:2]
 
    raw, dropped = collect_objects(sample, w, h)
    objects = resolve_overlaps(raw, w, h)
 
    scene = {
        "image": Path(sample.filepath).name,
        "image_size": [w, h],
        "objects": objects,
    }
    with open(out, "w") as f:
        json.dump(scene, f)
 
    overlay = str(Path(out).with_name(f"{Path(out).stem}_overlay.png"))
    save_overlay(sample.filepath, objects, overlay)
 
    print(f"  {len(raw)} objects read -> {len(objects)} after merging and removing overlaps")
    if dropped:
        listing = ", ".join(f"{label} {score:.2f}" for label, score in sorted(dropped))
        print(f"  {len(dropped)} left out for confidence below {THRESHOLD}: {listing}")
 
    return len(objects), overlay
 
 
# One colour per label, in OpenCV's blue-green-red order
PALETTE = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
           (189, 103, 148), (75, 86, 140), (194, 119, 227), (34, 189, 188)]
 
 
def save_overlay(image_path, objects, out, alpha=0.45):
    """Draws the exported objects on the background image.
 
    Drawn from the same objects as the JSON, so the picture shows exactly
    what was saved. Each name tag sits inside its own object.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    labels = sorted({o["label"] for o in objects})
    colour = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}
    masks = [object_mask(o, w, h) for o in objects]
 
    fill = image.copy()
    for o, mask in zip(objects, masks):
        fill[mask > 0] = colour[o["label"]]
    overlay = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
 
    # Largest first, so a door's outline is drawn on top of the wall around it
    order = sorted(range(len(objects)), key=lambda i: -int(masks[i].sum()))
    for o in (objects[i] for i in order):
        outlines = [np.array(o["polygon"], np.int32)]
        outlines += [np.array(hole, np.int32) for hole in o["holes"]]
        cv2.polylines(overlay, outlines, True, colour[o["label"]], 2)
 
    font = cv2.FONT_HERSHEY_SIMPLEX
    for o, mask in zip(objects, masks):
        # Tag at the point deepest inside the object
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        depth = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        cy, cx = np.unravel_index(int(np.argmax(depth)), depth.shape)
        (tw, th), _ = cv2.getTextSize(o["label"], font, 0.45, 1)
        x = int(min(max(cx - tw // 2 - 3, 0), w - tw - 6))
        y = int(min(max(cy - th // 2 - 3, 0), h - th - 6))
        cv2.rectangle(overlay, (x, y), (x + tw + 6, y + th + 6), colour[o["label"]], -1)
        cv2.putText(overlay, o["label"], (x + 3, y + th + 3), font, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
 
    cv2.imwrite(out, overlay)
 
 
# ----------------------------------------------------------------- main
 
def collect_datasets():
    """Loads the existing datasets and segments any videos not yet done."""
    datasets = []
 
    for name in EXISTING_DATASETS:
        if not fo.dataset_exists(name):
            raise RuntimeError(f"No dataset named '{name}'. Available: {fo.list_datasets()}")
        dataset = fo.load_dataset(name)
        dataset.persistent = True
        datasets.append(dataset)
 
    model = None
    for video in VIDEOS:
        name = f"scene_{Path(video).stem}"
        if fo.dataset_exists(name):
            print(f"Reusing existing dataset '{name}' (skipping SAM 3)")
            datasets.append(fo.load_dataset(name))
            continue
        background = median_background_over_a_video(video)
        print(f"Background saved to: {background}")
        print(f"Segmenting {background}")
        model = model or load_model()  # load SAM 3 only once
        datasets.append(segment_image(background, name, model))
 
    return datasets
 
 
def main():
    datasets = collect_datasets()
    if not datasets:
        raise SystemExit("Nothing to do: fill in EXISTING_DATASETS or VIDEOS")
 
    session = fo.launch_app(datasets[0])
    for i, dataset in enumerate(datasets):
        if i > 0:
            session.dataset = dataset  # show the next scene in the same App
        input(f"[{i + 1}/{len(datasets)}] '{dataset.name}': annotate in the App, "
              "then press Enter here to save... ")
 
        out = f"{dataset.name}.json"
        count, overlay = export_scene(dataset, out)
        print(f"Saved {count} objects to {out}, picture to {overlay}")
 
 
if __name__ == "__main__":
    main()
"""Segment the static scene of videos and export each as JSON.

Flow: video -> median background image -> SAM 3 masks -> hand annotation
in the FiftyOne App -> <dataset name>.json with pixel-coordinate polygons
and <dataset name>_overlay.png showing those polygons on the background.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

# Datasets that are already segmented: only opened, annotated and exported
EXISTING_DATASETS = []#"scene_seg", "scene_seg1"

# Videos still to segment: each becomes a dataset named scene_<video name>
VIDEOS = ["20251002_121145_id121_enter.mp4"]  # e.g. ["20250916_095917_id7_enter.mp4"]

PROMPTS = ["floor", "wall", "door", "window"]
THRESHOLD = 0.7  # minimum confidence for a SAM 3 object to be kept


def median_background_over_a_video(source, step=3):
    """Builds a person-free background image as the median over frames."""
    cap = cv2.VideoCapture(source)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {source}")

    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    name = f"{Path(source).stem}_background.png"
    cv2.imwrite(name, background)
    return name


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


def export_scene(dataset, out):
    """Writes SAM 3 masks and hand annotations to JSON as pixel polygons."""
    dataset.reload()  # pick up annotations made in the App
    sample = dataset.first()
    h, w = cv2.imread(sample.filepath).shape[:2]

    def px(points):
        return [[round(x * w), round(y * h)] for x, y in points]

    objects = []
    for field in sample.field_names:
        value = sample[field]

        if isinstance(value, fo.Detections):
            for det in value.detections:
                # Hand annotations have no confidence and are always kept
                if det.confidence is not None and det.confidence < THRESHOLD:
                    continue
                if det.mask is not None:  # SAM 3 mask
                    shapes = det.to_polyline(tolerance=2).points
                else:  # hand-drawn box
                    x, y, bw, bh = det.bounding_box
                    shapes = [[(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]]
                for shape in shapes:
                    objects.append({
                        "label": det.label.lower(),
                        "source": field,
                        "confidence": det.confidence,
                        "polygon": px(shape),
                    })

        elif isinstance(value, fo.Polylines):  # hand-drawn polygon
            for pl in value.polylines:
                for shape in pl.points:
                    objects.append({
                        "label": pl.label.lower(),
                        "source": field,
                        "confidence": None,
                        "polygon": px(shape),
                    })

    scene = {
        "image": Path(sample.filepath).name,
        "image_size": [w, h],
        "objects": objects,
    }
    with open(out, "w") as f:
        json.dump(scene, f, indent=2)

    overlay = str(Path(out).with_name(f"{Path(out).stem}_overlay.png"))
    save_overlay(sample.filepath, objects, overlay)

    return len(objects), overlay


# One colour per label, in OpenCV's blue-green-red order
PALETTE = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
           (189, 103, 148), (75, 86, 140), (194, 119, 227), (34, 189, 188)]


def save_overlay(image_path, objects, out, alpha=0.45):
    """Draws the exported polygons on the background image.

    Drawn from the same objects as the JSON, so the picture shows exactly
    what was saved.
    """
    image = cv2.imread(image_path)
    labels = sorted({o["label"] for o in objects})
    colour = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}

    fill = image.copy()
    for o in objects:
        pts = np.array(o["polygon"], dtype=np.int32)
        cv2.fillPoly(fill, [pts], colour[o["label"]])
    overlay = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for o in objects:
        pts = np.array(o["polygon"], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, colour[o["label"]], 2)
        x, y = (int(v) for v in pts.min(axis=0))
        (tw, th), _ = cv2.getTextSize(o["label"], font, 0.5, 1)
        cv2.rectangle(overlay, (x, y), (x + tw + 8, y + th + 8), colour[o["label"]], -1)
        cv2.putText(overlay, o["label"], (x + 4, y + th + 4), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out, overlay)


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
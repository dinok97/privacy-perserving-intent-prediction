"""
segment images using SAM 3 and export JSON and overlays

Flow:
1. Load SAM 3 model and processor once.
2. For each image:
   a. Preprocess the image once.
   b. Compute image features once.
   c. For each text prompt:
      i. Preprocess the text prompt.
      ii. Run the model to get segmentation outputs.
"""

from pathlib import Path
import json
import time
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


IMAGES = []
SEGMENTATION_DEST_DIR = (
    Path(__file__).resolve().parent / ".." / "data" / "segmentation_results"
).resolve()

# Confidence threshold for accepting a detected concept or object
# Adjust this value to control the number of detected objects.
THRESHOLD = 0.65

# Threshold applied to the predicted mask's pixel-level scores 
# to decide which pixels belong to the object.
MASK_THRESHOLD = 0.5

PROMPTS = ["floor", "wall", "door", "window"]

PALETTE = [
    (180, 119, 31), (14, 127, 255),
    (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140),
    (194, 119, 227), (34, 189, 188),
]

# Set to True to print debug information for each image and prompt.
PRINT_DEBUG = False


def load_model():
    """Load SAM 3 once for the entire batch."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 on {device}...")

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3")
    model = model.to(device).eval()

    print("SAM 3 loaded.")
    return model, processor


def mask_to_polygons(mask, tolerance=2):
    """Convert a binary mask into pixel-coordinate polygons."""
    mask = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue

        approx = cv2.approxPolyDP(
            contour, tolerance, True
        ).reshape(-1, 2)

        if len(approx) >= 3:
            polygons.append(approx.tolist())

    return polygons


def save_overlay(image_path, objects, out, alpha=0.45):
    """Draw the exported polygons on the original image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    labels = sorted({obj["label"] for obj in objects})
    colours = {
        label: PALETTE[i % len(PALETTE)]
        for i, label in enumerate(labels)
    }

    fill = image.copy()

    for obj in objects:
        pts = np.array(obj["polygon"], dtype=np.int32)
        cv2.fillPoly(fill, [pts], colours[obj["label"]])

    overlay = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj in objects:
        pts = np.array(obj["polygon"], dtype=np.int32)
        colour = colours[obj["label"]]

        cv2.polylines(overlay, [pts], True, colour, 2)

        x, y = pts.min(axis=0)
        x, y = int(x), int(y)

        label = obj["label"]
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)

        cv2.rectangle(
            overlay, (x, y), (x + tw + 8, y + th + 8),
            colour, -1
        )
        cv2.putText(
            overlay, label, (x + 4, y + th + 4),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    cv2.imwrite(str(out), overlay)


def segment_image(image_path, model, processor):
    """
    Segment one image using multiple prompts.

    The image is encoded once; its visual embeddings are reused
    for each text prompt.
    """
    image_path = Path(image_path)

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    width, height = image.size

    # 1. Preprocess the image once.
    image_inputs = processor(images=image, return_tensors="pt").to(model.device)

    # 2. Compute image features once.
    with torch.inference_mode():
        vision_embeds = model.get_vision_features(
            pixel_values=image_inputs.pixel_values
        )

    objects = []

    # 3. Run the four text prompts using the same image features.
    for prompt in PROMPTS:
        start = time.perf_counter()

        text_inputs = processor(
            text=prompt, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(
                vision_embeds=vision_embeds,
                **text_inputs
            )

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=[[height, width]],
        )[0]

        scores = results["scores"].detach().cpu().numpy()
        masks = results["masks"].detach().cpu().numpy()

        count = 0

        for score, mask in zip(scores, masks):
            # Convert the predicted mask to polygons.
            binary_mask = mask.squeeze().astype(bool)
            polygons = mask_to_polygons(binary_mask)

            for polygon in polygons:
                objects.append({
                    "label": prompt,
                    "source": prompt,
                    "confidence": float(score),
                    "polygon": polygon,
                })
                count += 1

        elapsed = time.perf_counter() - start
        if PRINT_DEBUG:
            print(f"  {prompt}: {count} polygons " f"({elapsed:.2f}s)")

    return {
        "image": image_path.name,
        "image_size": [width, height],
        "objects": objects,
    }


def segment_images(
    images: list[str],
    dest_dir: str = "",
    on_progress=None,
):
    """Segment images and export JSON and overlays without FiftyOne.

    on_progress, if given, is called after each image (ok=False for a skipped one)."""
    # Load the model only once.
    model, processor = load_model()

    total_start = time.perf_counter()

    for image_path in images:
        image_path = Path(image_path)

        if not image_path.is_file():
            print(f"Skipping missing image: {image_path}")
            if on_progress:
                on_progress(ok=False)
            continue

        start = time.perf_counter()
        print(f"Segmenting {image_path.name}...")

        scene = segment_image(image_path, model, processor)

        json_path = Path(image_path).with_name(f"{Path(image_path).stem}.json")
        overlay_path = Path(image_path).with_name(f"{Path(image_path).stem}_overlay.png")
            
        if dest_dir:    
            dest_path = Path(dest_dir)
            dest_path.mkdir(parents=True, exist_ok=True)
            json_path = dest_path / f"{image_path.stem}.json"
            overlay_path = (dest_path / f"{image_path.stem}_overlay.png")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2)

        save_overlay(image_path, scene["objects"], overlay_path)

        elapsed = time.perf_counter() - start

        if PRINT_DEBUG:
            print(f"  Objects: {len(scene['objects'])}")
            print(f"  JSON: {json_path}")
            print(f"  Overlay: {overlay_path}")
            print(f"  Image total: {elapsed:.2f}s")

        if on_progress:
            on_progress()

    print(
        f"\nAll done in "
        f"{time.perf_counter() - total_start:.2f}s"
    )


def main():
    images = sys.argv[1:] or IMAGES
    
    if not images:
        raise SystemExit("Nothing to do: fill in IMAGES or pass image paths")
    
    segment_images(images)


if __name__ == "__main__":
    main()
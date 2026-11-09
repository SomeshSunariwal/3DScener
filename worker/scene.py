"""Build the image assets consumed by the After Effects scene script."""

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from image_input import load_still
from inference import fill_background, infer_depth
from models import MODELS, model_path, valid_model
from mobile_sam import PromptSegmenter, load_selection_fallback, recover_uncovered_objects
from segmentation import segment_objects


def prepare_photo(source):
    image = load_still(source)
    image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    if min(image.size) < 32:
        raise ValueError("Choose an image at least 32 pixels wide and tall.")
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "black"), rgba)
    return image.convert("RGB")


def process(data, key, source, segmentation, fill, progress, selection=None, prompt_model="mobile-sam", job_id=None):
    manual = selection is not None
    for model, role in ((key, "depth"), (segmentation, "segmentation"), (fill, "fill")):
        if MODELS.get(model, {}).get("role") != role:
            raise ValueError(f"Choose a valid {role} model.")
        if not valid_model(data, model):
            raise ValueError(f"Download {MODELS[model]['label']} in Model Setup.")
    image = prepare_photo(source)
    occupied = np.zeros((image.height, image.width), dtype=bool)
    detections = []

    if manual:
        if not isinstance(selection, list) or not 1 <= len(selection) <= 12:
            raise ValueError("Add between 1 and 12 objects to the selection list.")
        segmenter = PromptSegmenter(data, image, prompt_model)
        progress(progress=f"Building the selected object masks with {MODELS[prompt_model]['label']}…")
        for index, object_selection in enumerate(selection, start=1):
            mask = segmenter.selection_mask(object_selection)
            box = mask.getbbox()
            x1, y1, x2, y2 = box
            pixels = np.asarray(mask.crop(box)) > 0
            occupied[y1:y2, x1:x2] |= pixels
            detections.append((box, pixels, index))

    model_label = MODELS[segmentation]["label"]
    progress(progress=f"Detecting remaining objects with {model_label}…" if manual else
             f"Separating objects with {model_label}…")
    # Detect on the original photo for context, then reserve the manual pixels.
    limit = 12 + len(detections)
    if manual and prompt_model == segmentation and segmenter.yolo:
        automatic = segmenter.detections[:limit]
    else:
        automatic = segment_objects(image, model_path(data, segmentation), limit=limit,
                                    architecture=MODELS[segmentation].get("architecture"))
    if MODELS[segmentation].get("architecture") == "yolo26-seg":
        covered = occupied.copy()
        for (left, top, right, bottom), pixels in automatic:
            covered[top:bottom, left:right] |= pixels
        remaining_area = occupied.size - occupied.sum()
        detected_area = covered.sum() - occupied.sum()
        if detected_area < .03 * remaining_area:
            progress(progress="Looking for subjects missed by YOLO26x-seg…")
            if manual:
                fallback = segmenter.fallback_segmenter() if segmenter.yolo else segmenter
            else:
                fallback = load_selection_fallback(data, image)
            if fallback:
                progress(progress=f"Recovering object masks with {fallback.label}…")
                automatic.extend(recover_uncovered_objects(fallback, covered, limit=min(5, limit)))
            elif not automatic:
                raise ValueError("YOLO26x-seg found no objects. Download HQ-SAM 2 in Model Setup to enable fallback masks.")
    for box, pixels in automatic:
        if manual:
            x1, y1, x2, y2 = box
            remaining = pixels & ~occupied[y1:y2, x1:x2]
            if remaining.sum() < max(16, pixels.sum() * .50):
                continue
            pixels = remaining
        detections.append((box, pixels, None))
    if not detections:
        raise ValueError("No clear objects were detected. Try a photo with distinct subjects.")

    progress(progress="Estimating depth from the full image with DirectML…")
    depth = infer_depth(image, model_path(data, key), MODELS[key]["nearIsHigh"])
    depth = depth.resize(image.size, Image.Resampling.BILINEAR)
    depth_pixels = np.asarray(depth)
    candidates = []
    for box, pixels, selection_index in detections:
        x1, y1, x2, y2 = box
        distance = float(np.median(depth_pixels[y1:y2, x1:x2][pixels])) / 255
        candidates.append((distance, box, pixels, selection_index))

    cutouts = []
    auto_count = 0
    for nearness, box, pixels, selection_index in sorted(candidates, key=lambda entry: entry[0], reverse=True):
        x1, y1, x2, y2 = box
        if selection_index is None:
            if manual and auto_count == 5:
                continue
            claimed = occupied[y1:y2, x1:x2]
            visible = pixels & ~claimed
            if visible.sum() < max(16, pixels.sum() * .20):
                continue
            pixels = visible
            claimed |= visible
            auto_count += 1
        cutouts.append((nearness, box, pixels, selection_index))
    if not cutouts or occupied.all():
        raise ValueError("Could not separate the objects from the background in this image.")

    progress(progress=f"Filling the space behind {len(cutouts)} objects with {MODELS[fill]['label']}…")
    holes = Image.fromarray(occupied.astype(np.uint8) * 255)
    background = fill_background(image, holes, model_path(data, fill), fill)

    progress(progress="Saving scene layers…")
    job = data / "scenes" / (job_id or uuid4().hex)
    job.mkdir(parents=True)
    background.save(job / "background.png")
    depth.save(job / "depth.png")
    layers = []
    auto_number = 0
    for index, (nearness, box, visible, selection_index) in enumerate(reversed(cutouts), start=1):
        layer = image.crop(box)
        layer.putalpha(Image.fromarray(visible.astype(np.uint8) * 255))
        path = job / f"object-{index:02}.png"
        layer.save(path)
        if selection_index is not None:
            name = f"Object {selection_index:02}"
        elif manual:
            auto_number += 1
            name = f"Auto Object {auto_number:02}"
        else:
            name = f"Object {index:02}"
        layers.append({"name": name, "file": str(path),
                       "x": box[0], "y": box[1], "nearness": nearness})
    return {
        "name": Path(source).stem,
        "width": image.width,
        "height": image.height,
        "layers": layers,
        "background": str(job / "background.png"),
        "depth": str(job / "depth.png"),
    }


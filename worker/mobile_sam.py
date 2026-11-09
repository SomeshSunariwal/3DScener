"""Object masks from point and freehand prompts using local SAM models."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageDraw

from models import MODELS, model_files, model_path, valid_model
from segmentation import detection_session, yolo_objects


class PromptSegmenter:
    def __init__(self, data, photo, key):
        import onnxruntime as ort

        model = MODELS.get(key, {})
        if model.get("role") != "prompt" and not model.get("selection"):
            raise ValueError("Choose a model for click selection.")
        if not valid_model(data, key):
            raise ValueError(f"Download {model['label']} in Model Setup.")
        self.size = photo.size
        self.label = model["label"]
        self.yolo = model.get("architecture") == "yolo26-seg"
        if self.yolo:
            self.width, self.height = self.size
            self.data = data
            self.photo = photo
            self.fallback = None
            self.detector = detection_session(model_path(data, key))
            self.detections = yolo_objects(photo, self.detector, limit=32)
            return
        encoder_path, decoder_path = (path for path, _ in model_files(data, key))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.decoder = ort.InferenceSession(str(decoder_path), options, providers=["CPUExecutionProvider"])
        self.hq = key == "hq-sam-2"
        if self.hq:
            self.width = self.height = 1024
            self.features = hq_image_features(photo, encoder_path, options, ort)
        else:
            scale = 1024 / max(photo.size)
            self.width, self.height = (max(1, round(side * scale)) for side in photo.size)
            resized = photo.resize((self.width, self.height), Image.Resampling.BILINEAR)
            encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
            embedding = encoder.run(None, {"input_image": np.asarray(resized, dtype=np.float32)})[0]
            self.features = {"image_embeddings": embedding}

    def _decode(self, points, box=None):
        if not isinstance(points, list) or not 1 <= len(points) <= 40:
            raise ValueError("Use between 1 and 40 selection points per object.")
        coords, labels = [], []
        for x, y, label in points:
            if not 0 <= x <= 1 or not 0 <= y <= 1 or label not in (0, 1):
                raise ValueError("A selection point is outside the photo.")
            coords.append([x * self.width, y * self.height])
            labels.append(label)
        if 1 not in labels:
            raise ValueError("Select an object before refining the mask.")
        if self.yolo:
            x, y = coords[labels.index(1)]
            target = np.zeros((self.height, self.width), dtype=bool)
            col = min(self.width - 1, int(x))
            row = min(self.height - 1, int(y))
            target[row, col] = True
            return self._detected_mask(target, point=(col, row))
        if box is None:
            coords.append([0, 0])
            labels.append(-1)
        else:
            left, top, right, bottom = box
            corners = [[left, top], [right, bottom]]
            if self.hq:
                coords[:0] = corners
                labels[:0] = [2, 3]
            else:
                coords.extend(corners)
                labels.extend((2, 3))
        inputs = {
            **self.features,
            "point_coords": np.array([coords], dtype=np.float32),
            "point_labels": np.array([labels], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
        }
        if self.hq:
            masks, scores = self.decoder.run(["masks", "iou_predictions"], inputs)
            logits = Image.fromarray(masks[0, scores[0].argmax()])
            return np.asarray(logits.resize((self.width, self.height), Image.Resampling.BILINEAR)) > 0
        inputs["orig_im_size"] = np.array([self.height, self.width], dtype=np.float32)
        masks, = self.decoder.run(["masks"], inputs)
        return masks[0, 0] > 0

    def _image_mask(self, pixels):
        mask = Image.fromarray(pixels.astype(np.uint8) * 255)
        mask = mask.resize(self.size, Image.Resampling.NEAREST)
        if not mask.getbbox():
            raise ValueError("No mask at that point. Try clicking inside the object.")
        return mask

    def selection_mask(self, selection):
        if isinstance(selection, list):
            return self._image_mask(self._decode(selection))
        edits = selection["edits"]
        if len(edits) > 64:
            raise ValueError("Use up to 64 freehand edits per mask.")
        if self.yolo and "outline" in selection:
            current = self._detected_mask(np.asarray(self._outline(selection["outline"])))
        else:
            current = self._decode([[*selection["point"], 1]])
        if not current.any():
            raise ValueError("No mask at that point. Try clicking inside the object.")
        for edit in edits:
            if edit["mode"] == "add" and "point" in edit:
                x, y = edit["point"]
                snapped = self._decode([[x, y, 1]])
                col = min(self.width - 1, int(x * self.width))
                row = min(self.height - 1, int(y * self.height))
                if self.yolo:
                    if not (snapped & ~current).any():
                        raise ValueError("That object is already selected.")
                elif current[row, col] or not snapped[row, col]:
                    raise ValueError("Click inside an unselected object to add it.")
                current |= snapped
                continue
            if edit["mode"] not in ("add", "remove"):
                raise ValueError("Draw a freehand shape around the area to change.")
            current = self._snap_edit(current, self._outline(edit["points"]), edit["mode"] == "remove")
        return self._image_mask(current)

    def _outline(self, points):
        if not 3 <= len(points) <= 512:
            raise ValueError("Draw a freehand shape around the area to change.")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
            raise ValueError("The freehand shape must be inside the image.")
        outline = Image.new("1", (self.width, self.height))
        polygon = [(round(x * (self.width - 1)), round(y * (self.height - 1))) for x, y in points]
        ImageDraw.Draw(outline).polygon(polygon, fill=1)
        return outline

    def _snap_edit(self, current, outline, remove):
        eligible = current if remove else ~current
        inside = np.asarray(outline)
        target = inside & eligible
        rows, cols = np.nonzero(target)
        if not rows.size:
            hint = "inside the current mask" if remove else "outside the current mask"
            raise ValueError(f"Draw around the object or detail {hint}.")
        if self.yolo:
            snapped = self._detected_mask(target, eligible)
            updated = current & ~snapped if remove else current | snapped
            if not updated.any():
                updated = current & ~self._fallback_mask(target, eligible)
                if not updated.any():
                    raise ValueError("Use Discard Mask to remove the entire selection.")
            return updated

        # The outline guides the decoder with a box and a point inside the area being changed.
        outline_rows, outline_cols = np.nonzero(inside)
        seed = np.argmin((rows - outline_rows.mean()) ** 2 + (cols - outline_cols.mean()) ** 2)
        y, x = rows[seed], cols[seed]
        prompt = [[x / self.width, y / self.height, 1]]
        left, top, right, bottom = outline.getbbox()
        box = (left, top, right - 1, bottom - 1)
        for bounds in (box, None):
            snapped = self._decode(prompt, bounds)
            changed = snapped & eligible
            if not snapped[y, x] or (changed & target).sum() < .10 * changed.sum():
                continue
            updated = current & ~snapped if remove else current | snapped
            if updated.any():
                return updated
        raise ValueError("Could not snap to that area. Draw more closely around the object or detail.")

    def _detected_mask(self, target, eligible=None, point=None):
        minimum_overlap = .05 if point is None else 0
        best = self._best_detection(self.detections, target, eligible, minimum_overlap)
        nearby = target
        if best is None and point:
            x, y = point
            radius = max(2, min(16, round(min(self.size) * .008)))
            nearby = target.copy()
            nearby[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = True
            best = self._best_detection(self.detections, nearby, eligible, minimum_overlap)
        if best is None:
            focused = self._focused_detections(target)
            best = self._best_detection(focused, target, eligible, minimum_overlap)
            if best is None and nearby is not target:
                best = self._best_detection(focused, nearby, eligible, minimum_overlap)
        if best is None:
            return self._fallback_mask(target, eligible, point)
        mask = np.zeros_like(target)
        region, pixels = best
        mask[region] = pixels
        return mask

    def _focused_detections(self, target):
        bounds = Image.fromarray(target).getbbox()
        if not bounds:
            return []
        left, top, right, bottom = bounds
        width = min(self.width, max(self.width // 3, (right - left) * 2))
        height = min(self.height, max(self.height // 3, (bottom - top) * 2))
        x = min(self.width - width, max(0, (left + right - width) // 2))
        y = min(self.height - height, max(0, (top + bottom - height) // 2))
        crop = self.photo.crop((x, y, x + width, y + height))
        detections = yolo_objects(crop, self.detector, limit=32, min_score=.18)
        return [((x + l, y + t, x + r, y + b), pixels) for (l, t, r, b), pixels in detections]

    def _best_detection(self, detections, target, eligible, minimum_overlap):
        target_area = target.sum()
        best = None
        best_overlap = 0
        for box, pixels in detections:
            x1, y1, x2, y2 = box
            region = np.s_[y1:y2, x1:x2]
            matches = (pixels & target[region]).sum()
            if not matches:
                continue
            area = pixels.sum() if eligible is None else (pixels & eligible[region]).sum()
            if eligible is not None and matches < .10 * area:
                continue
            overlap = matches / (area + target_area - matches)
            if overlap >= minimum_overlap and overlap > best_overlap:
                best = (region, pixels)
                best_overlap = overlap
        return best

    def fallback_segmenter(self):
        if self.fallback is None:
            self.fallback = load_selection_fallback(self.data, self.photo)
        return self.fallback

    def _fallback_mask(self, target, eligible, point=None):
        segmenter = self.fallback_segmenter()
        if segmenter is None:
            raise ValueError("YOLO26x-seg couldn't match this object. Download HQ-SAM 2 in Model Setup to enable fallback masks.")
        if point is not None:
            x, y = point
            pixels = segmenter.mask_at((x + .5) / self.width, (y + .5) / self.height)
            if pixels[y, x]:
                return pixels
        else:
            rows, cols = np.nonzero(target)
            seed = np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2)
            x, y = (cols[seed] + .5) / self.width, (rows[seed] + .5) / self.height
            left, top, right, bottom = Image.fromarray(target).getbbox()
            box = (left * segmenter.width / self.width, top * segmenter.height / self.height,
                   right * segmenter.width / self.width, bottom * segmenter.height / self.height)
            for bounds in (box, None):
                pixels = segmenter.mask_at(x, y, bounds)
                changed = pixels if eligible is None else pixels & eligible
                overlap = (changed & target).sum()
                if overlap and overlap >= .10 * changed.sum():
                    return pixels
        raise ValueError("Couldn't match a mask here. Try a click inside the object or a tighter outline.")

    def mask_at(self, x, y, box=None):
        mask = Image.fromarray(self._decode([[x, y, 1]], box))
        return np.array(mask.resize(self.size, Image.Resampling.NEAREST))


def load_selection_fallback(data, photo):
    for key in ("hq-sam-2", "mobile-sam"):
        if all(path.is_file() for path, _ in model_files(data, key)):
            return PromptSegmenter(data, photo, key)
    return None


def recover_uncovered_objects(segmenter, occupied, limit):
    height, width = occupied.shape
    candidates = []
    for y in (.5, .25, .75):
        for x in (.5, .25, .75):
            row, col = int(y * height), int(x * width)
            if occupied[row, col]:
                continue
            pixels = segmenter.mask_at(x, y)
            if not pixels[row, col]:
                continue
            area = pixels.sum()
            if not .01 * occupied.size <= area <= .65 * occupied.size:
                continue
            box = Image.fromarray(pixels).getbbox()
            left, top, right, bottom = box
            if sum((left == 0, top == 0, right == width, bottom == height)) >= 3:
                continue
            candidates.append((area, box, pixels[top:bottom, left:right].copy()))
    claimed = occupied.copy()
    objects = []
    for area, (left, top, right, bottom), pixels in sorted(candidates, key=lambda item: -item[0]):
        region = claimed[top:bottom, left:right]
        if (pixels & ~region).sum() < .65 * area:
            continue
        objects.append(((left, top, right, bottom), pixels))
        region |= pixels
        if len(objects) == limit:
            break
    return objects


def hq_image_features(photo, encoder_path, options, ort):
    resized = photo.resize((1024, 1024), Image.Resampling.BILINEAR)
    image_hash = hashlib.sha256(resized.tobytes()).hexdigest()
    cache_key = MODELS["hq-sam-2"]["sha256"] + image_hash
    cache = encoder_path.with_suffix(".features.npz")
    names = ["image_embed", "high_res_feats_0", "high_res_feats_1"]
    try:
        with np.load(cache, allow_pickle=False) as saved:
            if saved["key"].item() == cache_key:
                return {name: saved[name] for name in names}
    except (OSError, ValueError, KeyError, BadZipFile):
        pass

    image = np.asarray(resized, dtype=np.float32) / 255
    image -= np.array([.485, .456, .406], dtype=np.float32)
    image /= np.array([.229, .224, .225], dtype=np.float32)
    encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    inputs = {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None])}
    features = dict(zip(names, encoder.run(names, inputs)))
    # The CLI starts a new process for every gesture. Keep one image's features locally.
    temporary = None
    try:
        with NamedTemporaryFile(dir=cache.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, key=cache_key, **features)
        temporary.replace(cache)
    except OSError:
        # Continue with the computed features if the cache is unwritable or in use.
        if temporary:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return features

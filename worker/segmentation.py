"""Turn instance detections into masks in the original photo coordinates."""

from math import ceil, floor

import numpy as np
from PIL import Image


def detection_session(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def segment_objects(photo, model, limit=12, architecture=None):
    engine = detection_session(model)
    if architecture == "yolo26-seg":
        return yolo_objects(photo, engine, limit)
    return mask_rcnn_objects(photo, engine, limit)


def mask_rcnn_objects(photo, engine, limit):
    width, height = photo.size
    scale = min(800 / min(width, height), 1333 / max(width, height))
    resized = photo.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)
    bgr = rgb[:, :, ::-1] - np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    padded = np.zeros((3, ceil(resized.height / 32) * 32, ceil(resized.width / 32) * 32), dtype=np.float32)
    padded[:, :resized.height, :resized.width] = bgr.transpose(2, 0, 1)
    boxes, _, scores, masks = engine.run(None, {engine.get_inputs()[0].name: padded})

    objects = []
    sx, sy = width / resized.width, height / resized.height
    for index in np.argsort(-scores):
        if scores[index] < .55 or len(objects) == limit:
            break
        left, top, right, bottom = boxes[index]
        # Clip in image space but preserve the original box-to-mask mapping.
        x1, y1 = max(0, floor(left * sx)), max(0, floor(top * sy))
        x2, y2 = min(width, ceil((right + 1) * sx)), min(height, ceil((bottom + 1) * sy))
        if x2 <= x1 or y2 <= y1 or right <= left or bottom <= top:
            continue
        mask = Image.fromarray(masks[index, 0].astype(np.float32))
        region = ((x1 / sx - left) * mask.width / (right + 1 - left),
                  (y1 / sy - top) * mask.height / (bottom + 1 - top),
                  (x2 / sx - left) * mask.width / (right + 1 - left),
                  (y2 / sy - top) * mask.height / (bottom + 1 - top))
        mask = mask.transform((x2 - x1, y2 - y1), Image.Transform.EXTENT, region,
                              Image.Resampling.BILINEAR)
        pixels = np.asarray(mask) > .5
        if pixels.sum() >= max(16, width * height * .0005):
            objects.append(((x1, y1, x2, y2), pixels))
    return objects


def yolo_objects(photo, engine, limit, min_score=.35):
    width, height = photo.size
    input_spec = engine.get_inputs()[0]
    input_h, input_w = input_spec.shape[2:]
    scale = min(input_w / width, input_h / height)
    resized = photo.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                           Image.Resampling.BILINEAR)
    pad_x = (input_w - resized.width) // 2
    pad_y = (input_h - resized.height) // 2
    letterbox = Image.new("RGB", (input_w, input_h), (114, 114, 114))
    letterbox.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(letterbox, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    predictions, prototypes = engine.run(None, {input_spec.name: np.ascontiguousarray(tensor)})
    proto = prototypes[0]
    channels, proto_h, proto_w = proto.shape
    rows = predictions[0].T
    class_scores = rows[:, 4:-channels]
    labels = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    centers, radii = rows[:, :2], rows[:, 2:4] / 2
    boxes = np.concatenate((centers - radii, centers + radii), axis=1)

    sx, sy = width / resized.width, height / resized.height
    objects = []
    for index in yolo_indices(boxes, scores, labels, min_score):
        left, top, right, bottom = boxes[index]
        x1, y1 = max(0, floor((left - pad_x) * sx)), max(0, floor((top - pad_y) * sy))
        x2, y2 = min(width, ceil((right - pad_x) * sx)), min(height, ceil((bottom - pad_y) * sy))
        if x2 <= x1 or y2 <= y1:
            continue
        logits = rows[index, -channels:] @ proto.reshape(channels, -1)
        mask = Image.fromarray(logits.reshape(proto_h, proto_w))
        region = ((pad_x + x1 / sx) * proto_w / input_w,
                  (pad_y + y1 / sy) * proto_h / input_h,
                  (pad_x + x2 / sx) * proto_w / input_w,
                  (pad_y + y2 / sy) * proto_h / input_h)
        pixels = np.asarray(mask.transform((x2 - x1, y2 - y1), Image.Transform.EXTENT,
                                           region, Image.Resampling.BILINEAR)) > 0
        if pixels.sum() >= 16:
            objects.append(((x1, y1, x2, y2), pixels))
            if len(objects) == limit:
                break
    return objects


def yolo_indices(boxes, scores, labels, min_score=.35):
    """Keep high-confidence boxes using class-aware non-maximum suppression."""
    indices = np.flatnonzero(scores >= min_score)
    indices = indices[np.argsort(-scores[indices])]
    sizes = np.maximum(0, boxes[:, 2:] - boxes[:, :2])
    areas = sizes[:, 0] * sizes[:, 1]
    while indices.size:
        index = indices[0]
        yield index
        rest = indices[1:]
        starts = np.maximum(boxes[index, :2], boxes[rest, :2])
        ends = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        overlap = np.maximum(0, ends - starts)
        intersection = overlap[:, 0] * overlap[:, 1]
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        indices = rest[(labels[rest] != labels[index]) | (iou <= .7)]

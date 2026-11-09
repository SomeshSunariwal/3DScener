"""Image preparation, depth inference, and background fill."""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from runtime import gpu_session
from lama_graph import directml_lama


def tensor(image):
    return np.asarray(image, dtype=np.float32) / np.float32(255)

def infer_depth(photo, path, near_is_high):
    engine = gpu_session(str(path))
    input_spec = engine.get_inputs()[0]
    side = input_spec.shape[-2] if isinstance(input_spec.shape[-2], int) else 518
    resized = ImageOps.contain(photo, (side, side), Image.Resampling.BICUBIC)
    x = (side - resized.width) // 2
    y = (side - resized.height) // 2
    rgb = np.asarray(resized)
    padded = np.pad(rgb, ((y, side - resized.height - y),
                          (x, side - resized.width - x), (0, 0)), mode="edge")
    if input_spec.type == "tensor(uint8)":
        model_input = padded[None]
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (tensor(padded) - mean) / std
        dtype = np.float16 if input_spec.type == "tensor(float16)" else np.float32
        model_input = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=dtype)
    depth = np.squeeze(engine.run(None, {input_spec.name: model_input})[0])
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("The depth model returned an invalid depth map.")
    depth = Image.fromarray(depth.astype(np.float32)).resize((side, side), Image.Resampling.BILINEAR)
    depth = np.asarray(depth.crop((x, y, x + resized.width, y + resized.height)))
    near = depth if near_is_high else -depth
    low, high = np.percentile(near, [3, 97])
    if high - low <= 1e-7:
        raise ValueError("This photo has too little depth variation to separate into layers.")
    normalized = np.clip((near - low) / (high - low), 0, 1)
    return Image.fromarray(np.uint8(normalized * 255))

def fill_mat(image_input, mask_input, path):
    try:
        import torch
        from safetensors.torch import load_file
        from spandrel_extra_arches.architectures.MAT import MATArch
    except (ImportError, OSError) as error:
        raise RuntimeError("Install MAT support in Model Setup before processing.") from error

    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = MATArch().load(load_file(str(path))).to(dtype=torch.float32).eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(image_input), torch.from_numpy(mask_input))
    return output.numpy()[0] * 255


def fill_background(photo, mask, path, model="lama"):
    # Include a margin for a short pull-out and remove subject-edge remnants.
    px, py = (max(2, round(length * .07)) for length in photo.size)
    padded = ImageOps.expand(photo, border=(px, py))
    hole = ImageOps.expand(mask.filter(ImageFilter.MaxFilter(7)), border=(px, py), fill=255)
    if hole.getextrema() != (0, 255):
        raise ValueError("No distinct foreground was found. Try a photo with a clear subject.")
    width, height = padded.size
    fill_size = (max(1, round(width * 512 / max(width, height))),
                 max(1, round(height * 512 / max(width, height))))
    fill_image = padded.resize(fill_size, Image.Resampling.BICUBIC)
    fill_mask = hole.resize(fill_size, Image.Resampling.NEAREST)
    extra_x, extra_y = 512 - fill_size[0], 512 - fill_size[1]
    image_input = np.pad(tensor(fill_image), ((0, extra_y), (0, extra_x), (0, 0)), mode="edge")
    mask_input = np.pad(tensor(fill_mask), ((0, extra_y), (0, extra_x)), mode="edge")
    image_input = np.ascontiguousarray(image_input.transpose(2, 0, 1)[None])
    mask_input = np.ascontiguousarray(mask_input[None, None])
    if model == "mat":
        output = fill_mat(image_input, mask_input, path)
    else:
        engine = gpu_session(directml_lama(path))
        output = engine.run(None, {"image": image_input, "mask": mask_input})[0][0]
    fill = Image.fromarray(np.uint8(np.clip(output.transpose(1, 2, 0), 0, 255)))
    fill = fill.crop((0, 0, *fill_size)).resize(padded.size, Image.Resampling.BICUBIC)
    return Image.composite(fill, padded, hole)


"""JSON-lines command interface for the CEP panel."""

import json
import os
from pathlib import Path
import sys
from uuid import UUID

from models import MODELS, download, model_files, model_path, valid_model
from runtime import runtime_check


def emit(**event):
    print(json.dumps(event), flush=True)


def read_selection(value):
    return json.load(sys.stdin) if value == "-" else json.loads(value)


def check(data):
    from importlib.util import find_spec

    try:
        runtime_check()
        runtime_error = ""
    except Exception as error:
        runtime_error = str(error)
    emit(result={
        "runtime": not runtime_error,
        "runtimeError": runtime_error,
        "matRuntime": all(find_spec(package) is not None for package in (
            "torch", "torchvision", "spandrel", "spandrel_extra_arches", "safetensors"
        )),
        "models": {key: valid_model(data, key) for key in MODELS},
    })


def preview(filename):
    from base64 import b64encode
    from io import BytesIO
    from image_input import load_still

    thumbnail = load_still(filename)
    thumbnail.thumbnail((480, 300))
    output = BytesIO()
    thumbnail.convert("RGBA").save(output, format="PNG")
    return b64encode(output.getvalue()).decode("ascii")


def main():
    action, directory, *arguments = sys.argv[1:]
    data = Path(directory)
    if action == "check":
        check(data)
    elif action == "preview":
        emit(result=preview(arguments[0]))
    elif action == "mask":
        from base64 import b64encode
        from io import BytesIO
        from PIL import Image
        from mobile_sam import PromptSegmenter
        from scene import prepare_photo

        filename, selection, *prompt = arguments
        segmenter = PromptSegmenter(data, prepare_photo(filename), prompt[0] if prompt else "mobile-sam")
        mask = segmenter.selection_mask(read_selection(selection))
        mask.thumbnail((1024, 1024), Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", mask.size, (74, 212, 225))
        overlay.putalpha(mask.point(lambda pixel: pixel * .45))
        output = BytesIO()
        overlay.save(output, format="PNG")
        emit(result=b64encode(output.getvalue()).decode("ascii"))
    elif action == "download":
        download(data, arguments[0], emit)
        emit(result=True)
    elif action == "delete":
        key, = arguments
        if key not in MODELS:
            raise ValueError("Unknown model.")
        for path, _ in model_files(data, key):
            path.unlink(missing_ok=True)
        if key == "hq-sam-2":
            model_path(data, key).with_suffix(".features.npz").unlink(missing_ok=True)
        emit(result=True)
    elif action == "process" and MODELS.get(arguments[0], {}).get("role") == "depth":
        from scene import process

        key, filename, segmentation, fill, *selection = arguments
        prompts = read_selection(selection[0]) if selection else None
        prompt_model = selection[1] if len(selection) > 1 else "mobile-sam"
        job = os.environ.get("SCENER_JOB_ID")
        emit(result=process(data, key, filename, segmentation, fill, progress=emit,
                            selection=prompts, prompt_model=prompt_model, job_id=UUID(job).hex if job else None))
    else:
        raise ValueError("Unknown 3DScener command.")


if __name__ == "__main__":
    try:
        main()
    except Exception as failure:
        emit(error=str(failure) or type(failure).__name__)
        sys.exit(1)

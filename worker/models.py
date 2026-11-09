"""Model registry, verified downloads, and local asset paths."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


CATALOG = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))
MODELS = {key: entry for key, entry in CATALOG.items() if "url" in entry}


def model_path(data, key):
    return data / "models" / (key + MODELS[key].get("extension", ".onnx"))


def model_files(data, key):
    model = MODELS[key]
    path = model_path(data, key)
    yield path, model
    if "companion" in model:
        yield path.with_name(key + "-decoder.onnx"), model["companion"]


def valid_file(path, checksum):
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == checksum


def valid_model(data, key):
    return all(valid_file(path, entry["sha256"]) for path, entry in model_files(data, key))


def download(data, key, progress):
    files = list(model_files(data, key))
    total = sum(entry.get("bytes", 0) for _, entry in files)
    completed = 0

    def report(**event):
        if total:
            event["received"] += completed
            event["total"] = total
        progress(**event)

    for target, model in files:
        download_file(target, model, report)
        completed += model.get("bytes", 0)


def download_file(target, model, progress):
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, model["sha256"]):
        return
    request = Request("https://huggingface.co/" + model["url"], headers={"User-Agent": "3DScener/1.0"})
    digest = hashlib.sha256()
    temporary = None
    try:
        with urlopen(request, timeout=45) as response:
            total = int(response.headers.get("Content-Length", 0))
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                received = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    received += len(block)
                    progress(progress="Downloading model", received=received, total=total)
        if digest.hexdigest() != model["sha256"]:
            raise ValueError("The download did not match its checksum. Try downloading it again.")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


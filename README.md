# 3DScener

3DScener turns detected objects in a still photo into editable 3D layers in After Effects. It places the cutouts at different depths, fills the background, and animates a camera to show the parallax.

## Install on Windows

Requires x64 Windows, After Effects 25 or newer with CEP 12, a DirectX 12 compatible GPU, and 64-bit CPython 3.11–3.13. Install Python from [python.org](https://www.python.org/downloads/windows/) if needed.

1. Close After Effects.
2. From PowerShell in the 3DScener folder, run:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-3DScener.ps1
   ```

   The installer copies the panel into your user CEP extensions folder and enables CEP 12 developer mode for unsigned panels in your Windows account.
3. Open After Effects and select **Window > Extensions > 3DScener**.

## Make a scene

1. Open **Model Setup**. Install the GPU runtime and download a depth model, a fill model, and an **Object Segmentation Model** for automatic detection. Choose Mask R-CNN or **YOLO26x-seg**. Under **Selection Models**, download **HQ-SAM 2 (Precise Selection)** for detailed masks, **MobileSAM (Click Selection)** for faster selection, or **YOLO26x-seg** to select detected objects. The two YOLO entries share a single download. Choose the active model in the panel’s **Selection Model** dropdown before starting a mask.
2. Drop an image into the panel or click **Browse files**. Choose a depth model and fill model, then click **Process Image** to open the selection toolbar.
   Hold **Ctrl** and scroll over the image to zoom around the pointer. Hold the middle mouse button and drag to pan while zoomed in. Scroll down with **Ctrl** to return to the fitted view.
   Choose **Select Object**, then click an object to snap to its outline. With **Add**, click another object or drag a freehand outline around the area to include. Use **Remove** to draw around an area to exclude. Freehand edits snap when you release. Click **Add to List** to save the mask. To change a saved mask, click **Edit**, refine it, and click **Update in List**. **Process Selected** keeps your saved masks and detects up to five more objects in the remaining area. Choose **Auto Only** to detect objects across the entire photo. **Stop Processing** cancels the current job and closes the selection tools.
3. Click **Create AE Layer**. 3DScener opens a new composition sized to the processed image with separate object layers, a filled background, an animated camera, and a hidden depth map.
4. Play the timeline to see the camera slide. Edit the camera's **Position** keyframes to adjust the motion.

The setup downloads packages from PyPI and weights from Hugging Face. Photo processing runs locally. Models and the isolated Python runtime live in `%LOCALAPPDATA%\3DScener`. Generated PNGs live in its `scenes` folder. After Effects references those files, so keep them in place or use **File > Dependencies > Collect Files** before moving a project.

## Processing details

- ONNX Runtime uses DirectML first for both depth estimation and LaMa. Unsupported operators can run on the CPU, as do image decoding and layer preparation. A session that cannot initialize DirectML reports an error.
- Mask R-CNN runs on the CPU and creates up to 12 cutouts from the objects it recognizes. Depth Anything estimates depth from the full photo so all cutouts share the same scale. Each cutout is placed at the median depth of its mask. Choose LaMa or MAT to fill the space behind the cutouts and extend the frame edges. MAT runs on the CPU and installs its support packages on first download.
- The local runtime reads raster formats supported by its Pillow installation, including JPEG, PNG, TIFF, WebP, BMP, and GIF. The first frame of an animation or the first page of a multipage image is used. The panel generates a PNG preview for formats other than JPEG and PNG.
- Images are capped at 4096 pixels on the long edge. Transparent inputs are composited over black. Scene layers and the depth map are saved as PNG.

## Code layout

- `index.html`, `panel.css`, and `panel.js` render the panel and manage its workflow.
- `local-runtime.js` locates Python and communicates with the worker. `host/scene.jsx` builds the composition in After Effects.
- `worker/models.json` is the shared model catalog. `models.py` manages local weights and downloads. A new model with a compatible ONNX input and output can be registered in the catalog. Other model architectures need their own inference implementation.
- `worker/runtime.py` initializes and probes DirectML. `segmentation.py` runs Mask R-CNN, `inference.py` runs depth and fill, and `scene.py` writes the layer assets. `lama_graph.py` adapts the LaMa export's Fourier products to work with DirectML.

## Model and runtime sources

- [Mask R-CNN Full](https://huggingface.co/onnxmodelzoo/MaskRCNN-12) and [Mask R-CNN Compact](https://huggingface.co/onnxmodelzoo/MaskRCNN-12-qdq)
- [MobileSAM ONNX exports](https://huggingface.co/Acly/MobileSAM)
- [HQ-SAM 2](https://github.com/SysCV/sam-hq/tree/main/sam-hq2), [ONNX encoder](https://huggingface.co/latolukasz/sam2.1-hiera-large-onnx), and [ONNX decoder](https://huggingface.co/latolukasz/sam2.1-hq-hiera-large-onnx)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) and [YOLO26x-seg ONNX export](https://huggingface.co/unileon-robotics/YOLO26-ONNX/blob/e046fc5125ee6b579f1801f0b6df68a60b949c42/yolo26x-seg.onnx) (AGPL-3.0)
- [Depth Anything V2 Small ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-small)
- [Depth Anything V2 Base ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-base)
- [Depth Anything V2 Large ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-large)
- [Depth Anything V3 Small and Base ONNX exports](https://huggingface.co/zerochocobo/DepthAnything3_ONNX)
- [Depth Anything V3 Mono Large ONNX export](https://huggingface.co/Jens-Duttke/Depth-Anything-3-MONO-ONNX)
- [OpenCV LaMa ONNX export](https://huggingface.co/opencv/inpainting_lama)
- [MAT Places512 weights](https://huggingface.co/Acly/MAT) and [MAT noncommercial license](https://github.com/fenglinglwb/MAT/blob/main/LICENSE)
- [ONNX Runtime DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [Adobe CEP 12](https://github.com/Adobe-CEP/CEP-Resources/tree/master/CEP_12.x)

"""DirectML session configuration and runtime checks."""

import os


def runtime_check():
    import numpy as np
    from PIL import Image
    from onnx import TensorProto, helper
    import onnxruntime

    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("DirectML is missing. Install the GPU runtime from Model Setup.")
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "DirectML probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
    gpu_session(model.SerializeToString()).run(None, {"x": np.array([-1, 1], dtype=np.float32)})


def gpu_session(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(6, os.cpu_count() or 1)
    options.log_severity_level = 3
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    engine = ort.InferenceSession(
        model, options,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    if engine.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError("DirectML could not start. Check the GPU driver and GPU memory.")
    engine.disable_fallback()
    return engine


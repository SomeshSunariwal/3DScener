"""Adapt LaMa's batched Fourier products to DirectML's supported tensor ranks."""

import onnx
from onnx import TensorProto, helper


def directml_lama(path):
    model = onnx.shape_inference.infer_shapes(onnx.load(path))
    graph = model.graph
    ranks = {item.name: len(item.type.tensor_type.shape.dim)
             for item in [*graph.value_info, *graph.input]}
    ranks.update({item.name: len(item.dims) for item in graph.initializer})

    prefix = "scener_dml_"
    for label, value in (("zero", 0), ("one", 1), ("last", -1), ("matrix", -2), ("end", 2**63 - 1)):
        graph.initializer.append(helper.make_tensor(prefix + label, TensorProto.INT64, [1], [value]))

    nodes = []
    for index, node in enumerate(graph.node):
        if (node.op_type != "MatMul" or ranks.get(node.input[0]) != 2
                or ranks.get(node.input[1]) != 5):
            nodes.append(node)
            continue

        # [M,K] × [B,C,D,K,N] becomes [M,K] × [B*C*D,K,N].
        # Restore the batch axes after the equivalent lower-rank multiplication.
        left, right = node.input
        name = prefix + str(index) + "_"
        nodes.extend([
            helper.make_node("Shape", [left], [name + "left_shape"]),
            helper.make_node("Shape", [right], [name + "right_shape"]),
            helper.make_node("Slice", [name + "left_shape", prefix + "zero", prefix + "one"], [name + "rows"]),
            helper.make_node("Slice", [name + "right_shape", prefix + "matrix", prefix + "end"], [name + "matrix_shape"]),
            helper.make_node("Slice", [name + "right_shape", prefix + "zero", prefix + "matrix"], [name + "batch_shape"]),
            helper.make_node("Slice", [name + "right_shape", prefix + "last", prefix + "end"], [name + "cols"]),
            helper.make_node("Concat", [prefix + "last", name + "matrix_shape"], [name + "flat_shape"], axis=0),
            helper.make_node("Reshape", [right, name + "flat_shape"], [name + "flat_right"]),
            helper.make_node("MatMul", [left, name + "flat_right"], [name + "product"]),
            helper.make_node("Concat", [name + "batch_shape", name + "rows", name + "cols"], [name + "output_shape"], axis=0),
            helper.make_node("Reshape", [name + "product", name + "output_shape"], list(node.output)),
        ])
    del graph.node[:]
    graph.node.extend(nodes)
    return model.SerializeToString()

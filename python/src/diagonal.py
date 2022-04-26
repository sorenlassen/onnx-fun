import string
import numpy as np
import onnx
import onnxruntime


# onnx helpers
Numpy2OnnxType = {
        np.float32: onnx.TensorProto.FLOAT,
        np.float64: onnx.TensorProto.DOUBLE,
        np.int32: onnx.TensorProto.INT32,
        np.int64: onnx.TensorProto.INT64,
    }

def param(param_name, dtype, shape):
    onnx_dtype = Numpy2OnnxType[dtype.type if isinstance(dtype, np.dtype) else dtype]
    return onnx.helper.make_tensor_value_info(
            param_name,
            onnx_dtype,
            shape)

def run_model(model, output_names, input_names, inputs):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    return sess.run(output_names, dict(zip(input_names, inputs)))


# diagonal helpers:

# Checks input args. Returns non-negative axes and the output shape.
def diagonal_check_arguments(data, offset=0, axis1=0, axis2=1):
    assert data.ndim >= 2
    assert all(-data.ndim <= x < data.ndim for x in [axis1, axis2])
    if axis1 < 0: axis1 += data.ndim
    if axis2 < 0: axis2 += data.ndim
    assert axis1 != axis2, f"{axis1},{axis2}"
    dim = data.shape[axis1]
    assert dim == data.shape[axis2]
    offset_dim = dim - abs(offset) if -dim < offset < dim else 0
    oshape = list(data.shape) + [offset_dim]
    for ax in reversed(sorted([axis1, axis2])):
        del oshape[ax]
    return axis1, axis2, tuple(oshape)


# einsum helpers
ASCII_LETTERS = string.ascii_uppercase + string.ascii_lowercase

def einsum_diagonal_equation(axis1, axis2):
    assert axis1 >= 0 and axis2 >= 0 and axis1 != axis2
    assert max(axis1, axis2) <= len(ASCII_LETTERS), \
            f"this implementation only takes axes <= {len(ASCII_LETTERS)}"
    diag = ASCII_LETTERS[-1]
    other = ASCII_LETTERS[:max(axis1, axis2) - 1]
    inlst = list(other)
    for ax in sorted([axis1, axis2]):
        inlst.insert(ax, diag)
    return f"{''.join(inlst)}...->{other}...{diag}"

# Implements np.diagonal (without offset) with onnx Einsum.
def diagonal_by_einsum(data, offset=0, axis1=0, axis2=1):
    assert offset == 0, \
            "this implementation only takes offset 0"
    axis1, axis2, oshape = diagonal_check_arguments(data, axis1=axis1, axis2=axis2)
    equation = einsum_diagonal_equation(axis1, axis2)
    einsum_node = onnx.helper.make_node(
            "Einsum",
            inputs=["data"],
            outputs=["diagonalized"],
            equation=equation)
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_einsum',
                nodes=[einsum_node],
                inputs=[param("data", data.dtype, data.shape)],
                outputs=[param("diagonalized", data.dtype, oshape)],
                )
            )
    onnx.checker.check_model(model)
    [diagonalized] = run_model(model, ["diagonalized"], ["data"], [data])
    assert data.dtype == diagonalized.dtype
    assert oshape == diagonalized.shape
    return diagonalized

# Implements np.diagonal with onnx Slice and Einsum.
def diagonal_by_slice_einsum(data, offset=0, axis1=0, axis2=1):
    if offset == 0:
        return diagonal_by_einsum(data, 0, axis1, axis2)
    axis1, axis2, oshape = diagonal_check_arguments(data, offset, axis1, axis2)
    equation = einsum_diagonal_equation(axis1, axis2)
    axes = [axis1, axis2]
    dim = data.shape[axis1]
    if offset < 0:
        starts, ends = ([-offset, 0], [dim, offset])
    else:
        starts, ends = [0, offset], [-offset, dim]
    slice_node = onnx.helper.make_node(
            "Slice",
            inputs=["data", "starts", "ends", "axes"],
            outputs=["sliced"])
    einsum_node = onnx.helper.make_node(
            "Einsum",
            inputs=["sliced"],
            outputs=["diagonalized"],
            equation=equation)
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_slice_einsum',
                nodes=[slice_node, einsum_node],
                inputs=[
                    param("data", data.dtype, data.shape),
                    param("starts", np.int64, (2,)),
                    param("ends", np.int64, (2,)),
                    param("axes", np.int64, (2,)),
                    ],
                outputs=[param("diagonalized", data.dtype, oshape)],
                )
            )
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    [diagonalized] = run_model(model, ["diagonalized"],
            ["data", "starts", "ends", "axes"], [data, starts, ends, axes])
    assert data.dtype == diagonalized.dtype
    assert oshape == diagonalized.shape
    return diagonalized

def diagonal_test():
    print("diagonal_test() start")

    for data, axis1, axis2 in [
            (np.random.rand(2,2), 0, 1),
            (np.random.rand(2,2), 1, 0),
            (np.random.rand(2,2), -1, -2),
            (np.random.rand(1,2,2,3), 1, 2),
            (np.random.rand(2,1,3,2), 0, 3),
            ]:
        expected = np.diagonal(data, axis1=axis1, axis2=axis2)
        actual1 = diagonal_by_einsum(data, axis1=axis1, axis2=axis2)
        actual2 = diagonal_by_slice_einsum(data, axis1=axis1, axis2=axis2)
        assert expected.dtype == actual1.dtype == actual2.dtype
        assert expected.shape == actual1.shape == actual2.shape
        assert np.allclose(expected, actual1)
        assert np.allclose(expected, actual2)

    print("diagonal_test() end")

if __name__ == "__main__":
   diagonal_test()

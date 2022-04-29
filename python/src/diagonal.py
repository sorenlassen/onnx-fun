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

def onnx_type(dtype):
    return Numpy2OnnxType[dtype.type if isinstance(dtype, np.dtype) else dtype]

def param(param_name, dtype, shape):
    return onnx.helper.make_tensor_value_info(
            param_name,
            onnx_type(dtype),
            shape)

def make_constant_node(name, tensor):
    return onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor(
            name=name,
            data_type=onnx_type(tensor.dtype),
            dims=tensor.shape,
            vals=tensor.flatten(),
        ),
    )
def run_model(model, *inputs):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    def names(params): return map(lambda param: param.name, params)
    inputs_dict = dict(zip(names(model.graph.input), inputs))
    output_names = list(names(model.graph.output))
    return sess.run(output_names, inputs_dict)


# diagonal helpers:

# Checks input args. Returns non-negative axes and the output shape.
def diagonal_check_arguments(data_shape, offset, axis1, axis2):
    ndim = len(data_shape)
    assert ndim >= 2
    assert all(-ndim <= x < ndim for x in [axis1, axis2])
    if axis1 < 0: axis1 += ndim
    if axis2 < 0: axis2 += ndim
    assert axis1 != axis2, f"{axis1},{axis2}"
    dim = data_shape[axis1]
    assert dim == data_shape[axis2]
    offset_dim = dim - abs(offset) if -dim < offset < dim else 0
    oshape = list(data_shape) + [offset_dim]
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
    assert offset == 0, "this implementation only takes offset 0"
    axis1, axis2, oshape = diagonal_check_arguments(data.shape, offset, axis1, axis2)
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
    [diagonalized] = run_model(model, data)
    assert data.dtype == diagonalized.dtype
    assert oshape == diagonalized.shape
    return diagonalized

# Implements np.diagonal with onnx Slice and Einsum.
def diagonal_by_slice_einsum(data, offset=0, axis1=0, axis2=1):
    if offset == 0:
        return diagonal_by_einsum(data, 0, axis1, axis2)
    axis1, axis2, oshape = diagonal_check_arguments(data.shape, offset, axis1, axis2)
    equation = einsum_diagonal_equation(axis1, axis2)
    axes = [axis1, axis2]
    dim = data.shape[axis1]
    if offset < 0:
        starts, ends = ([-offset, 0], [dim, offset])
    else:
        starts, ends = [0, offset], [-offset, dim]
    starts_node = make_constant_node("starts", starts)
    ends_node = make_constant_node("ends", ends)
    axes_node = make_constant_node("axes", axes)
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
                nodes=[starts_node, ends_node, axes_node, slice_node, einsum_node],
                inputs=[param("data", data.dtype, data.shape)],
                outputs=[param("diagonalized", data.dtype, oshape)],
                )
            )
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    [diagonalized] = run_model(model, data)
    assert data.dtype == diagonalized.dtype
    assert oshape == diagonalized.shape
    return diagonalized

# Implements np.diagonal with onnx GatherElements.
def diagonal_by_gather_elements_squeeze(data, offset=0, axis1=0, axis2=1):
    assert offset == 0, "this implementation only takes offset 0"
    axis1, axis2, oshape = diagonal_check_arguments(data.shape, offset, axis1, axis2)

    if {axis1, axis2} != {data.ndim - 2, data.ndim - 1}:
        # TODO: do this transpose in onnx
        perm = list(range(len(data.shape)))
        axis1, axis2 = sorted([axis1, axis2])
        del perm[axis2]
        del perm[axis1]
        perm += [axis1, axis2]
        data = data.transpose(perm)
        axis1, axis2 = data.ndim - 2, data.ndim - 1

    assert {axis1, axis2} == {data.ndim - 2, data.ndim - 1}, \
            "this implementation only works on the last axes"
    dim = oshape[-1]
    assert dim == data.shape[axis1] == data.shape[axis2]
    indices = np.broadcast_to(np.arange(dim), oshape).reshape(oshape + (1,))
    indices_node = make_constant_node("indices", indices)
    gather_elements_node = onnx.helper.make_node(
            "GatherElements",
            inputs=["data", "indices"],
            outputs=["gathered"],
            axis=-1)
    squeeze_axes_node = make_constant_node("squeeze_axes", np.array([-1]))
    squeeze_node = onnx.helper.make_node(
            "Squeeze",
            inputs=["gathered", "squeeze_axes"],
            outputs=["diagonalized"])
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_gather_elements',
                nodes=[indices_node, gather_elements_node, squeeze_axes_node, squeeze_node],
                inputs=[param("data", data.dtype, data.shape)],
                outputs=[param("diagonalized", data.dtype, oshape)],
                )
            )
    onnx.checker.check_model(model)
    [diagonalized] = run_model(model, data)
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
        actual3 = diagonal_by_gather_elements_squeeze(data, axis1=axis1, axis2=axis2)
        assert expected.dtype == actual1.dtype == actual2.dtype == actual3.dtype
        assert expected.shape == actual1.shape == actual2.shape == actual3.shape
        assert np.allclose(expected, actual1)
        assert np.allclose(expected, actual2)
        assert np.allclose(expected, actual3)

    print("diagonal_test() end")

if __name__ == "__main__":
   diagonal_test()

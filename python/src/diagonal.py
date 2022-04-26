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
    onnx_dtype = Numpy2OnnxType[dtype.type]
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
    assert data.shape[axis1] == data.shape[axis2]
    oshape = list(data.shape) + [data.shape[axis1]]
    for ax in reversed(sorted([axis1, axis2])):
        del oshape[ax]
    return axis1, axis2, tuple(oshape)


# einsum helpers
ASCII_LETTERS = list(string.ascii_uppercase + string.ascii_lowercase)

# Implements np.diagonal (without offset) with onnx.
def diagonal_by_einsum(data, offset=0, axis1=0, axis2=1):
    assert offset == 0
    axis1, axis2, oshape = diagonal_check_arguments(data, axis1=axis1, axis2=axis2)
    assert data.ndim <= len(ASCII_LETTERS), \
            "ran out of letters for indices, shape is too long"
    osubscripts = ASCII_LETTERS[:data.ndim - 1]
    isubscripts = osubscripts[:-1]
    for ax in sorted([axis1, axis2]):
        isubscripts.insert(ax, osubscripts[-1])
    equation = f"{''.join(isubscripts)}->{''.join(osubscripts)}"
    node = onnx.helper.make_node(
            "Einsum",
            inputs=["data"],
            outputs=["diag"],
            equation=equation)
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_einsum',
                nodes=[node],
                inputs=[param(node.input[0], data.dtype, data.shape)],
                outputs=[param(node.output[0], data.dtype, oshape)],
                )
            )
    onnx.checker.check_model(model)
    [diag] = run_model(model, node.output, node.input, [data])
    assert data.dtype == diag.dtype
    assert oshape == diag.shape
    return diag

def diagonal_by_einsum_test():
    print("diagonal_by_einsum_test() start")

    for data, axis1, axis2 in [
            (np.random.rand(2,2), 0, 1),
            (np.random.rand(2,2), 1, 0),
            (np.random.rand(2,2), -1, -2),
            (np.random.rand(1,2,2,3), 1, 2),
            (np.random.rand(2,1,3,2), 0, 3),
            ]:
        expected = np.diagonal(data, axis1=axis1, axis2=axis2)
        actual = diagonal_by_einsum(data, axis1=axis1, axis2=axis2)
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape, f"{expected.shape},{actual.shape}"
        assert np.allclose(expected, actual)

    print("diagonal_by_einsum_test() end")

if __name__ == "__main__":
   diagonal_by_einsum_test()

import string
import numpy as np
import onnx
import onnxruntime

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

ASCII_LETTERS = list(string.ascii_uppercase + string.ascii_lowercase)

# Implements np.diagonal (without offset) with onnx.
def diagonal_by_einsum(data, axis1=0, axis2=1):
    assert data.ndim >= 2
    assert all(-data.ndim <= x < data.ndim for x in [axis1, axis2])
    if axis1 < 0: axis1 += data.ndim
    if axis2 < 0: axis2 += data.ndim
    assert data.shape[axis1] == data.shape[axis2]
    assert axis1 != axis2
    fst, snd = min(axis1, axis2), max(axis1, axis2)
    assert data.ndim <= len(ASCII_LETTERS), \
            "ran out of letters for indices, shape is too long"
    osubscripts = ASCII_LETTERS[:data.ndim - 1]
    isubscripts = osubscripts[:-1]
    isubscripts.insert(fst, osubscripts[-1])
    isubscripts.insert(snd, osubscripts[-1])
    equation = f"{''.join(isubscripts)}->{''.join(osubscripts)}"
    node = onnx.helper.make_node(
            "Einsum",
            inputs=["data"],
            outputs=["diag"],
            equation=equation)
    in_shape = data.shape
    out_shape = in_shape[:fst] + in_shape[fst + 1:snd] + in_shape[snd + 1:] + (in_shape[fst],)
    dtype = data.dtype
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_einsum',
                nodes=[node],
                inputs=[param(node.input[0], dtype, in_shape)],
                outputs=[param(node.output[0], dtype, out_shape)],
                )
            )
    onnx.checker.check_model(model)
    [diag] = run_model(model, node.output, node.input, [data])
    assert data.dtype == diag.dtype
    assert out_shape == diag.shape
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
        actual = diagonal_by_einsum(data, axis1, axis2)
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape, f"{expected.shape},{actual.shape}"
        assert np.allclose(expected, actual)

    print("diagonal_by_einsum_test() end")

if __name__ == "__main__":
   diagonal_by_einsum_test()

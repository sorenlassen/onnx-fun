import string
import numpy as np
import onnx
import onnxruntime


# onnx helpers
def onnx_type(dtype):
    '''Returns equivalent onnx.TensorProto basetype for a given numpy type
    where dtype can be either a numpy dtype or np.float32, np.int64, etc.'''
    if isinstance(dtype, np.dtype): dtype = dtype.type
    return {
        np.float32: onnx.TensorProto.FLOAT,
        np.float64: onnx.TensorProto.DOUBLE,
        np.int32: onnx.TensorProto.INT32,
        np.int64: onnx.TensorProto.INT64,
    }[dtype]

def param(param_name, dtype, shape):
    return onnx.helper.make_tensor_value_info(
            param_name,
            onnx_type(dtype),
            shape)

def run_model(model, *inputs):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    def names(params): return map(lambda param: param.name, params)
    # model might omit an input, e.g. when result is just a constant
    assert len(model.graph.input) <= len(inputs)
    inputs_dict = dict(zip(names(model.graph.input), inputs))
    output_names = list(names(model.graph.output))
    return sess.run(output_names, inputs_dict)

def infer_shapes_and_run_model(model, *inputs):
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return run_model(model, *inputs)


def einsum_model(equation, ishapes, oshape):
    input_names = [f"x{i}" for i in range(len(ishapes))]
    output_name = "result"
    dtype = np.float64
    einsum_node = onnx.helper.make_node(
            "Einsum",
            inputs=input_names,
            outputs=[output_name],
            equation=equation)
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='einsum',
                nodes=[einsum_node],
                inputs=[
                        param(name, dtype, shape)
                        for name, shape in zip(input_names, ishapes)
                    ],
                outputs=[param(output_name, dtype, oshape)],
                )
            )

def einsum_model_test():
    print("einsum_model_test() start")

    for equation, ishapes, oshape in [
            ("ii->i", [(2,2)], (2,)),
            ("ij,jk", [(2,2),(2,2)], (2,2)),
            ]:
        inputs = [ np.random.rand(*shape) for shape in ishapes ]
        expected = np.einsum(equation, *inputs)
        model = einsum_model(equation, ishapes, oshape)
        [actual] = infer_shapes_and_run_model(model, *inputs)
        assert expected.shape == actual.shape
        np.testing.assert_almost_equal(expected, actual)

    print("einsum_model_test() end")

if __name__ == "__main__":
   einsum_model_test()

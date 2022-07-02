import numpy as np
import onnx
import onnxruntime

def onnx_type(dtype):
    '''Returns equivalent onnx.TensorProto.DataType for a given NumPy dtype
    where dtype can be either a np.dtype or np.float32, np.int64, etc.'''
    ty = np.dtype(dtype) # np.dtype() is idempotent
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty]

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
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return run_model(model, *inputs)

def onnex_model(op_type, *inputs, shape=None, dtype=None, **attributes):
    if dtype is None:
        dtype = inputs[0].dtype if inputs else np.float64
    input_names = [f"input{i}" for i in range(len(inputs))]
    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=input_names,
        outputs=["output"],
        **attributes)
    inputs = [param(n, t.dtype, t.shape) for n, t in zip(input_names, inputs)]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name=op_type,
        inputs=inputs,
        outputs=[param("output", dtype, shape)])
    return onnx.helper.make_model(graph=graph)

def onnex(op_type, *inputs, shape=None, dtype=None, **attributes):
    model = onnex_model(op_type, *inputs, shape=shape, dtype=dtype, **attributes)
    [result] = infer_shapes_and_run_model(model, *inputs)
    return result

def onnex_test():
    print("onnex_test() start")

    a2 = np.array([1.1,1.2])
    np.testing.assert_equal(a2, onnex("Identity", a2))

    r423 = np.random.rand(4,2,3)
    r5132 = np.random.rand(5,1,3,2)
    np.testing.assert_almost_equal(r423 @ r5132, \
        onnex("MatMul", r423, r5132))

    np.testing.assert_almost_equal(r423 @ r5132, \
        onnex("Einsum", r423, r5132, equation="abc,dace->dabe"))

    print("onnex_test() end")

if __name__ == "__main__":
   onnex_test()

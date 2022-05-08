import numpy as np
import onnx
import onnxruntime

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
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return run_model(model, *inputs)

def input_name(i):
    return f"input{i}"

def input_param(i, tensor):
    return param(input_name(i), tensor.dtype, tensor.shape)

def onix(opname, oshape, *inputs, dtype=None, **attributes):
    if dtype is None:
        dtype = inputs[0].dtype if len(inputs) > 0 else np.float64
    node = onnx.helper.make_node(
        opname,
        inputs=[input_name(i) for i, _ in enumerate(inputs)],
        outputs=["output"],
        **attributes)
    graph = onnx.helper.make_graph(
        [node],
        inputs=[input_param(i, t) for i, t in enumerate(inputs)],
        outputs=[param("output", dtype, oshape)],
        name=opname)
    model = onnx.helper.make_model(graph=graph)
    [result] = infer_shapes_and_run_model(model, *inputs)
    return result

def onix_test():
    print("onix_test() start")

    a2 = np.array([1.1,1.2])
    np.testing.assert_equal(a2, onix("Identity", (2,), a2))

    r423 = np.random.rand(4,2,3)
    r5132 = np.random.rand(5,1,3,2)
    np.testing.assert_almost_equal(r423 @ r5132, \
        onix("MatMul", (5,4,2,2), r423, r5132))

    np.testing.assert_almost_equal(r423 @ r5132, \
        onix("Einsum", (5,4,2,2), r423, r5132, equation="abc,dace->dabe"))

    print("onix_test() end")

if __name__ == "__main__":
   onix_test()
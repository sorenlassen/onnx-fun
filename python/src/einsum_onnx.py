import string
import numpy as np
import onnx
import onnxruntime
import einsum


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

def make_constant_node(output_name, tensor):
    tensor = np.asarray(tensor)
    return onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        value=onnx.helper.make_tensor(
            name=output_name,
            data_type=onnx_type(tensor.dtype),
            dims=tensor.shape,
            vals=tensor.flatten(),
        ),
    )


def make_constant_model(graph_name, output_name, tensor):
    constant_node = make_constant_node(output_name, tensor)
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name=graph_name,
                nodes=[constant_node],
                inputs=[],
                outputs=[param(output_name, tensor.dtype, tensor.shape)],
                )
            )

def make_identity_model(graph_name, input_name, output_name, dtype, shape):
    identity_node = onnx.helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[output_name],
    )
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name=graph_name,
                nodes=[identity_node],
                inputs=[param(input_name, dtype, shape)],
                outputs=[param(output_name, dtype, shape)],
                )
            )

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


def einsum_direct_model(equation, ishapes, dtype):
    spec = einsum.einsum_spec(equation, ishapes)
    oshape = spec.output.shape
    input_names = [f"x{i}" for i in range(len(ishapes))]
    output_name = "result"
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

def einsum_direct_model_test():
    print("einsum_direct_model_test() start")

    for equation, ishapes in [
            ("ii->i", [(2,2)]),
            ("ij,jk", [(2,2),(2,2)]),
            ]:
        inputs = [ np.random.rand(*shape) for shape in ishapes ]
        expected = np.einsum(equation, *inputs)
        model = einsum_direct_model(equation, ishapes, np.float64)
        [actual] = infer_shapes_and_run_model(model, *inputs)
        assert expected.shape == actual.shape
        np.testing.assert_almost_equal(expected, actual)

    print("einsum_direct_model_test() end")


def einsum_decomposed_model(equation, ishapes, dtype):
    spec = einsum.einsum_spec(equation, ishapes)
    oshape = spec.output.shape
    input_names = [f"x{i}" for i in range(len(ishapes))]
    output_name = "result"

    # In two cases the output is just zeros or empty:
    # (1) empty if there are any 0 dims in the output shape,
    # (2) zeros if there are any 0 dims in any input shape
    # (because they either occur in the output, which would be
    # empty, or will be eliminated by ReduceSum and become zeros).
    if any(np.prod(shape) == 0 for shape in ishapes + [oshape]):
        tensor = np.zeros(oshape, dtype=dtype)
        return make_constant_model('einsum_constant', output_name, tensor)

    if spec.is_identity():
        # The equation is the identity transformation on a single input.
        return make_identity_model(
                'einsum_identity', input_names[0], output_name,
                dtype, oshape)

    # TODO cover all the other cases...

def einsum_decomposed_model_test():
    print("einsum_decomposed_model_test() start")

    for equation, ishapes in [
            ("ii->i", [(0,0)]),
            ("ii", [(0,0)]),
            ("ij,jk", [(0,2),(2,2)]),
            ("ij,jk->k", [(0,2),(2,2)]),
            ("i", [(2,)]),
            ("...", [(2,3,4)]),
            ("ij...k->...ijk", [(2,3,4)]),
            ]:
        inputs = [ np.random.rand(*shape) for shape in ishapes ]
        expected = np.einsum(equation, *inputs)
        model = einsum_decomposed_model(equation, ishapes, np.float64)
        [actual] = infer_shapes_and_run_model(model, *inputs)
        assert expected.shape == actual.shape
        np.testing.assert_almost_equal(expected, actual)

    print("einsum_decomposed_model_test() end")

if __name__ == "__main__":
   einsum_direct_model_test()
   einsum_decomposed_model_test()

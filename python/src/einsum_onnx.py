import string
import numpy as np
import onnx
import onnxruntime
import einsum


# EinsumSpec helpers
def einsum_is_identity_spec(spec):
    if len(spec.inputs) != 1:
        return False
    if spec.inputs[0].idxs != spec.output.idxs:
        return False
    assert spec.inputs[0].shape == spec.output.shape
    return True


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

def nonneg(axes, ndim):
    if isinstance(axes, int):
        assert -ndim <= axes < ndim
        return axes if axes >= 0 else axes + ndim
    return list(map(lambda a: nonneg(a, ndim), axes))

def squeeze_shape(ishape, axes):
    oshape = list(ishape)
    for a in sorted(nonneg(axes, len(ishape)), reverse=True):
        del oshape[a]
    return tuple(oshape)

def make_squeeze_model(graph_name, input_name, output_name, dtype, shape, axes):
    axes_node = make_constant_node(graph_name, np.array(axes, dtype=np.int64))
    squeeze_node = onnx.helper.make_node(
        "Squeeze",
        inputs=[input_name, graph_name],
        outputs=[output_name],
    )
    oshape = squeeze_shape(shape, axes)
    model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name=graph_name,
                nodes=[axes_node, squeeze_node],
                inputs=[param(input_name, dtype, shape)],
                outputs=[param(output_name, dtype, oshape)],
                )
            )
    return model, oshape

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
    ninputs = len(ishapes)
    input_names = [f"in{i}" for i in range(ninputs)]

    # In two cases the output is just zeros or empty:
    # (1) empty if there are any 0 dims in the output shape,
    # (2) zeros if there are any 0 dims in any input shape
    # (because they either occur in the output, which would be
    # empty, or will be eliminated by ReduceSum and become zeros).
    oshape = spec.output.shape
    if any(np.prod(shape) == 0 for shape in ishapes + [oshape]):
        tensor = np.zeros(oshape, dtype=dtype)
        return make_constant_model('einsum_constant', "out", tensor)

    # Each transform is either an onnx model transforming the input
    # at that position or just a string with the name of the input
    # which represents the identity transformation.
    transforms = list(input_names)

    for i in range(ninputs):
        spec, transforms = einsum_diagonalize_input(spec, transforms, i, dtype)
        spec, transforms = einsum_reducesum_input(spec, transforms, i, dtype)

    while len(transforms) > 1:
        spec, transforms = einsum_contract_inputs(spec, transforms, 0, 1, dtype)

    return einsum_finalize(spec, transforms[0], dtype)

def einsum_diagonalize_input(spec, transforms, i, dtype):
    # TODO: implement
    return spec, transforms

def einsum_reducesum_input(spec, transforms, i, dtype):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    assert len(idxs) == len(set(idxs)), \
            "duplicates indexes after diagonalization pass"
    idxs_in_other_inputs = [
            idx in spec.inputs[j].idxs
            for j in range(len(spec.inputs)) if j != i
            ]
    idxs_only_in_i = set(idxs) - set(idxs_in_other_inputs + spec.output.idxs)

    # Squeeze any idxs only in inputs[i] that have dim 1.
    one_axes = []
    one_idxs = []
    for idx in idxs_only_in_i:
        axis = idxs.index(idx)
        if shape[axis] == 1:
            one_axes.append(axis)
            one_idxs.append(idx)
    transforms[i], shape = squeeze_output(transforms[i], one_axes, shape, dtype)

    # ReduceSum the rest.
    for idx in idxs_only_in_i - set(one_idxs):
        axis = idxs.index(idx)
        assert shape[axis] > 1
        transforms[i], shape = reducesum_output(transforms[i], axis, shape, dtype)
        del idxs[axis]

    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def transform_input_name(transform):
    if isinstance(transform, str):
        return transform
    else:
        return transform.graph.input[0].name

def transform_output_name(transform):
    if isinstance(transform, str):
        return transform
    else:
        return transform.graph.output[0].name

def squeeze_output(transform, axes, shape, dtype):
    if len(axes) == 0:
        return transform, shape
    graph_name = "squeeze" # TODO: make this unique
    iname = transform_input_name(transform)
    squeeze_model, oshape = make_squeeze_model(
            graph_name, iname, "out", dtype, shape, axes)
    if isinstance(transform, str):
        transform = squeeze_model
    else:
        oname = transform_output_name(transform)
        transform = onnx.compose.merge_models(
                transform, squeeze_model,
                io_map=[(oname, iname)])
    return transform, oshape

def reducesum_output(transform, axis, shape, dtype):
    # TODO: implement
    oshape = list(shape)
    del oshape[axis]
    return transform, oshape

def einsum_contract_inputs(spec, transforms, i, j, dtype):
    # TODO: implement
    return spec, transforms

def einsum_finalize(spec, transform, dtype):
    if einsum_is_identity_spec(spec):
        # The equation is the identity transformation.
        if isinstance(transform, str):
            return make_identity_model(
                    'einsum_identity', transform, "out",
                    dtype, spec.output.shape)
        return transform
    # TODO: transpose and/or reshape
    return transform


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
            # squeezes axes s,t,u:
            ("sij->ij", [(1,2,3)]),
            ("isj->ij", [(2,1,3)]),
            ("ijs->ij", [(2,3,1)]),
            ("sitju->ij", [(1,2,1,3,1)]),
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

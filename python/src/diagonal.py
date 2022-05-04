import string
import numpy as np
import onnx
import onnxruntime


# numpy helpers
def mutate_shape(shape, mutation_dict):
    lst = list(shape)
    for axis, dim in mutation_dict.items():
        lst[axis] = dim
    return tuple(lst)

def perm_from_dict(ndim, perm_dict):
    '''perm_dict maps axes to axes, each axis in [-ndim...ndim)'''
    assert all(-ndim <= k < ndim and -ndim <= v < ndim for k, v in perm_dict.items())
    def nonneg(a): return ndim + a if a < 0 else a
    d = {nonneg(k): nonneg(v) for k,v in perm_dict.items()}
    assert len(perm_dict) == len(d) == len(set(d.values()))
    perm = list(range(ndim))
    for a in sorted(d, reverse=True):
        del perm[a]
    for k, v in sorted(d.items(), key=lambda kv:kv[1]):
        perm.insert(v, k)
    return tuple(perm)

def transpose_by_dict(tensor, perm_dict):
    return tensor.transpose(perm_from_dict(tensor.ndim, perm_dict))


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

def make_empty_model(model_name, output_name, dtype, shape):
    empty = np.empty(shape, dtype=dtype)
    empty_node = make_constant_node(output_name, empty)
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name=model_name,
                nodes=[empty_node],
                inputs=[],
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
    oshape = list(data_shape)
    for ax in sorted([axis1, axis2], reverse=True):
        del oshape[ax]
    offset_dim = dim - abs(offset) if -dim < offset < dim else 0
    oshape.append(offset_dim)
    return axis1, axis2, tuple(oshape)

def diagonal_slice(dtype, ishape, offset, axis1, axis2):
    axis1, axis2, oshape = diagonal_check_arguments(ishape, offset, axis1, axis2)
    input_name = "diagonal_slice_input"
    output_name = "diagonal_slice_output"
    if offset == 0:
        sliced_shape = ishape
        slice_nodes = [
                onnx.helper.make_node(
                    "Identity",
                    inputs=[input_name],
                    outputs=[output_name]),
                ]
    else:
        sliced_dim = oshape[-1]
        sliced_shape = mutate_shape(ishape, {axis1: sliced_dim, axis2: sliced_dim})
        axes = [axis1, axis2]
        dim = ishape[axis1]
        if offset < 0:
            starts, ends = ([-offset, 0], [dim, offset])
        else:
            starts, ends = [0, offset], [-offset, dim]
        slice_nodes = [
            make_constant_node("starts", starts),
            make_constant_node("ends", ends),
            make_constant_node("axes", axes),
            onnx.helper.make_node(
                    "Slice",
                    inputs=[input_name, "starts", "ends", "axes"],
                    outputs=[output_name]),
            ]
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_slice',
                nodes=slice_nodes,
                inputs=[param(input_name, dtype, ishape)],
                outputs=[param(output_name, dtype, sliced_shape)],
                )
            )


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


# Implements np.diagonal with onnx Slice and Einsum.
def diagonal_by_slice_einsum(dtype, ishape, offset=0, axis1=0, axis2=1):
    axis1, axis2, oshape = diagonal_check_arguments(ishape, offset, axis1, axis2)
    if np.prod(oshape) == 0:
        return make_empty_model("diagonal_empty", "empty", dtype, oshape)
    # TODO: reuse diagonal_slice, same way it's used in
    # diagonal_by_gather_elements_squeeze_transpose
    if offset == 0:
        slice_output_name = "data"
        slice_nodes = []
    else:
        slice_output_name = "sliced"
        axes = [axis1, axis2]
        dim = ishape[axis1]
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
                outputs=[slice_output_name])
        slice_nodes = [starts_node, ends_node, axes_node, slice_node]
    equation = einsum_diagonal_equation(axis1, axis2)
    einsum_node = onnx.helper.make_node(
            "Einsum",
            inputs=[slice_output_name],
            outputs=["diagonalized"],
            equation=equation)
    return onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_slice_einsum',
                nodes=slice_nodes + [einsum_node],
                inputs=[param("data", dtype, ishape)],
                outputs=[param("diagonalized", dtype, oshape)],
                )
            )

# Implements np.diagonal with onnx GatherElements.
def diagonal_by_gather_elements(dtype, ishape, offset=0, axis1=0, axis2=1):
    axis1, axis2, oshape = diagonal_check_arguments(ishape, offset, axis1, axis2)
    if np.prod(oshape) == 0: # optimization
        return make_empty_model("diagonal_empty", "empty", dtype, oshape)
    slice_model = diagonal_slice(dtype, ishape, offset, axis1, axis2)
    odim = oshape[-1]
    sliced_shape = mutate_shape(ishape, {axis1: odim, axis2: odim})
    axis1, axis2 = sorted([axis1, axis2])
    indices_shape = sliced_shape[:axis1] + (1,) + sliced_shape[axis1 + 1:]
    arange = np.arange(odim)
    # same as arange.reshape((odim,) + (1,) * (len(sliced_shape) - 1 - axis2))
    unsqueezed = np.expand_dims(arange, tuple(range(1, len(sliced_shape) - axis2)))
    indices = np.broadcast_to(unsqueezed, indices_shape)
    indices_node = make_constant_node("indices", indices)
    # TODO: to optimize, if odim is 1, Reshape instead of GatherElements+Squeeze
    gather_elements_node = onnx.helper.make_node(
            "GatherElements",
            inputs=["data", "indices"],
            outputs=["gathered"],
            axis=axis1)
    squeeze_axes_node = make_constant_node("squeeze_axes", np.array([axis1]))
    squeeze_node = onnx.helper.make_node(
            "Squeeze",
            inputs=["gathered", "squeeze_axes"],
            outputs=["squeezed"])
    nodes = [indices_node, gather_elements_node, squeeze_axes_node, squeeze_node]
    if axis2 == len(sliced_shape) - 1:
        output_name = "squeezed"
    else:
        perm = perm_from_dict(len(oshape), {axis1: -1})
        transpose_node = onnx.helper.make_node(
                "Transpose",
                inputs=["squeezed"],
                outputs=["transposed"],
                perm=perm)
        nodes.append(transpose_node)
        output_name = "transposed"
    diag_model = onnx.helper.make_model(
            graph=onnx.helper.make_graph(
                name='diagonal_by_gather_elements',
                nodes=nodes,
                inputs=[param("data", dtype, sliced_shape)],
                outputs=[param(output_name, dtype, oshape)],
                )
            )
    [slice_output] = slice_model.graph.output
    [diag_input] = diag_model.graph.input
    return onnx.compose.merge_models(
            slice_model, diag_model,
            io_map=[(slice_output.name, diag_input.name)]
            )

def diagonal_test():
    print("diagonal_test() start")

    for shape, axis1, axis2 in [
            ((2,2), 0, 1),
            ((2,2), 1, 0),
            ((2,2), -1, -2),
            ((1,2,2,3), 1, 2),
            ((2,1,3,2), 0, 3),
            ((2,0,3,2), 0, 3),
            ((0,1,3,0), 0, 3),
            ]:
        for offset in (0, 1, 2):
            data = np.random.rand(*shape)
            dtype = data.dtype
            expected = np.diagonal(data, offset, axis1, axis2)
            models = [
                    diagonal_by_slice_einsum(dtype, shape, offset, axis1, axis2),
                    diagonal_by_gather_elements(dtype, shape, offset, axis1, axis2),
                ]
            for model in models:
                [actual] = infer_shapes_and_run_model(model, data)
                assert expected.dtype == actual.dtype
                assert expected.shape == actual.shape
                assert np.allclose(expected, actual)

    print("diagonal_test() end")

if __name__ == "__main__":
   diagonal_test()

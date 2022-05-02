from dataclasses import dataclass
from typing import List, Tuple, Union
import string
import numpy as np
import onnx # type: ignore
import onnxruntime # type: ignore
import einsum # type: ignore


# EinsumSpec helpers
def einsum_is_identity_spec(spec):
    if len(spec.inputs) != 1:
        return False
    if spec.inputs[0].idxs != spec.output.idxs:
        return False
    assert spec.inputs[0].shape == spec.output.shape
    return True

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

def make_identity_node(input_name, output_name):
    return onnx.helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=[output_name],
        )

def make_identity_model(graph_name, input_name, output_name, dtype, shape):
    identity_node = make_identity_node(input_name, output_name)
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


@dataclass
class Transform:
    iname: str
    ishape: Tuple[int, ...]
    dtype: Union[np.dtype, type] # e.g. np.type(np.int32) or np.int32
    oname: str
    oshape: Tuple[int, ...]
    nodes: List[onnx.onnx_ml_pb2.NodeProto]

    def graph(self, graph_name):
        if len(self.nodes) == 0:
            # Empty graphs don't come compose with onnx.compose, so
            # we insert an Identity node for robustness.
            final_oname = f"{graph_name}/out"
            final_nodes = [make_identity_node(self.iname, final_oname)]
        else:
            final_oname = self.oname
            final_nodes = self.nodes
        return onnx.helper.make_graph(
                name=graph_name,
                nodes=final_nodes,
                inputs=[param(self.iname, self.dtype, self.ishape)],
                outputs=[param(final_oname, self.dtype, self.oshape)],
                )

    def model(self, graph_name):
        return onnx.helper.make_model(graph = self.graph(graph_name))

    def next_name(self, stem):
        return f"{self.iname}/{stem}{len(self.nodes)}"

    def squeeze(self, axes):
        if len(axes) == 0:
            return self
        axes_tensor = np.array(axes, dtype=np.int64)
        axes_name = self.next_name("axes")
        self.nodes.append(make_constant_node(axes_name, axes_tensor))
        squeeze_name = self.next_name("squeeze")
        self.nodes.append(onnx.helper.make_node(
            "Squeeze",
            inputs=[self.oname, axes_name],
            outputs=[squeeze_name],
        ))
        self.oname = squeeze_name
        self.oshape = squeeze_shape(self.oshape, axes)
        return self

    def reducesum(self, axes):
        if len(axes) == 0:
            return self
        axes_tensor = np.array(axes, dtype=np.int64)
        axes_name = self.next_name("axes")
        self.nodes.append(make_constant_node(axes_name, axes_tensor))
        sum_name = self.next_name("sum")
        self.nodes.append(onnx.helper.make_node(
            "ReduceSum",
            inputs=[self.oname, axes_name],
            outputs=[sum_name],
            keepdims=0,
        ))
        self.oname = sum_name
        self.oshape = squeeze_shape(self.oshape, axes)
        return self

def make_identity_transform(dtype, shape, iname):
    return Transform(iname, shape, dtype, iname, shape, [])


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
    ninputs = len(ishapes)
    transforms = [
            make_identity_transform(dtype, ishapes[i], f"in{i}")
            for i in range(ninputs)
            ]

    for i in range(ninputs):
        spec, transforms = einsum_diagonalize_input(spec, transforms, i)
        spec, transforms = einsum_reducesum_input(spec, transforms, i)

    while len(transforms) > 1:
        spec, transforms = einsum_contract_inputs(spec, transforms, 0, 1)

    transform = einsum_finalize(spec, transforms[0])
    #print("final transform",transform)
    return transform.model(f"einsum({equation})")

def einsum_diagonalize_input(spec, transforms, i):
    # TODO: implement
    return spec, transforms

def einsum_reducesum_input(spec, transforms, i):
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
    transforms[i].squeeze(one_axes)
    for a in sorted(one_axes, reverse=True):
        del idxs[a]
        del shape[a]
    assert tuple(shape) == transforms[i].oshape

    # ReduceSum the rest.
    for idx in idxs_only_in_i - set(one_idxs):
        axis = idxs.index(idx)
        assert shape[axis] > 1
        transforms[i].reducesum([axis])
        del idxs[axis]
        del shape[axis]
        assert tuple(shape) == transforms[i].oshape

    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_contract_inputs(spec, transforms, i, j, dtype):
    # TODO: implement
    return spec, transforms

def einsum_finalize(spec, transform):
    if einsum_is_identity_spec(spec):
        # The equation is the identity transformation.
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
            # reducesum axes s,t,u:
            ("sij->ij", [(4,2,3)]),
            ("isj->ij", [(2,4,3)]),
            ("ijs->ij", [(2,3,4)]),
            ("sitju->ij", [(4,2,5,3,6)]),
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

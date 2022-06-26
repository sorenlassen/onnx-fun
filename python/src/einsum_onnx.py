from __future__ import annotations
from typing import Dict, List, Tuple, Union, Any
import os
import math
import numpy as np
import onnx # type: ignore
import onnxruntime # type: ignore
import einsum # type: ignore


# list/tuple helpers
def nonneg(pos, length: int):
    if isinstance(pos, int):
        assert -length <= pos < length
        return pos if pos >= 0 else pos + length
    return map(lambda p: nonneg(p, length), pos)

def omit(seq, *positions):
    for p in sorted(nonneg(positions, len(seq)), reverse=True):
        seq = seq[:p] + seq[p+1:]
    return seq


# EinsumSpec helpers
Shape = Tuple[int, ...]

def einsum_is_identity_spec(spec: einsum.EinsumSpec):
    if len(spec.inputs) != 1:
        return False
    if spec.inputs[0].idxs != spec.output.idxs:
        return False
    assert spec.inputs[0].shape == spec.output.shape, f"{spec}"
    return True

def shape_size(shape) -> int:
    return math.prod(shape)

def squeeze_shape(ishape, axes):
    oshape = list(ishape)
    for a in sorted(nonneg(axes, len(ishape)), reverse=True):
        del oshape[a]
    return tuple(oshape)

def unsqueeze_shape(ishape, axes):
    oshape = list(ishape)
    for a in sorted(nonneg(axes, len(ishape) + len(axes))):
        oshape.insert(a, 1)
    return tuple(oshape)

def transpose_seq(ishape, perm):
    return tuple(ishape[a] for a in perm)

def transpose_perm(original_seq, transposed_seq):
    assert sorted(original_seq) == sorted(transposed_seq)
    perm = tuple(original_seq.index(x) for x in transposed_seq)
    assert tuple(transposed_seq) == transpose_seq(original_seq, perm)
    return perm


# onnx helpers
def onnx_type(dtype : Union[np.dtype, type]) -> onnx.TensorProto.DataType:
    '''Returns equivalent onnx.TensorProto.DataType for a given NumPy dtype
    where dtype can be either a np.dtype or np.float32, np.int64, etc.'''
    ty = np.dtype(dtype) # np.dtype() is idempotent
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty]

def param(param_name, dtype, shape) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(
        param_name,
        onnx_type(dtype),
        shape)

def make_constant_node(output_name, tensor) -> onnx.NodeProto:
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

def make_typed_graph(graph_name, nodes, inputs, outputs, dtype) -> onnx.GraphProto:
    return onnx.helper.make_graph(
        name=graph_name,
        nodes=nodes,
        inputs=[param(name, dtype, shape) for name, shape in inputs],
        outputs=[param(name, dtype, shape) for name, shape in outputs],
    )

def run_model(model: onnx.ModelProto, *inputs):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    def names(params): return map(lambda param: param.name, params)
    # model might omit an input, e.g. when result is just a constant
    assert len(model.graph.input) <= len(inputs)
    inputs_dict = dict(zip(names(model.graph.input), inputs))
    output_names = list(names(model.graph.output))
    return sess.run(output_names, inputs_dict)

def infer_shapes_and_run_model(model: onnx.ModelProto, *inputs):
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return run_model(model, *inputs)


class Transform:
    name: str
    dtype: Union[np.dtype, type] # e.g. np.type(np.int32) or np.int32
    inputs: Dict[str, Shape] # maps input names to shapes
    oname: str
    oshape: Shape
    nodes: List[onnx.NodeProto]

    def __init__(self, name, shape, dtype):
        '''The identity transform from one input to the same output.'''
        self.name = name
        self.dtype = dtype
        self.inputs = {name: shape}
        self.oname = name
        self.oshape = shape
        self.nodes = []

    def graph(self, graph_name: str) -> onnx.GraphProto:
        if len(self.nodes) == 0:
            # Empty graphs don't compose with onnx.compose, so
            # we insert an Identity node for robustness.
            self.identity()
            assert len(self.nodes) == 1
        graph_inputs = sorted(self.inputs.items())
        graph_outputs = [(self.oname, self.oshape)]
        return make_typed_graph(graph_name, self.nodes, graph_inputs, graph_outputs, self.dtype)

    def model(self, graph_name: str) -> onnx.ModelProto:
        graph = self.graph(graph_name)
        return onnx.helper.make_model(graph)

    def next_name(self, stem: str):
        return f"{self.name}_{stem}_{len(self.nodes)}"

    def identity(self):
        identity_name = self.next_name("identity")
        self.nodes.append(onnx.helper.make_node(
            "Identity",
            inputs=[self.oname],
            outputs=[identity_name],
        ))
        self.oname = identity_name
        return self

    def reshape(self, shape: Shape):
        # cannot handle -1 dim in shape because we need to know the new oshape
        assert all(d >= 0 for d in shape), "no support for -1"
        if shape == self.oshape:
            return self
        shape_tensor = np.array(shape, dtype=np.int64)
        shape_name = self.next_name("shape")
        self.nodes.append(make_constant_node(shape_name, shape_tensor))
        reshape_name = self.next_name("reshape")
        self.nodes.append(onnx.helper.make_node(
            "Reshape",
            inputs=[self.oname, shape_name],
            outputs=[reshape_name],
        ))
        self.oname = reshape_name
        self.oshape = shape
        return self

    def squeeze(self, axes: List[int]):
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

    def unsqueeze(self, axes: List[int]):
        if len(axes) == 0:
            return self
        axes_tensor = np.array(axes, dtype=np.int64)
        axes_name = self.next_name("axes")
        self.nodes.append(make_constant_node(axes_name, axes_tensor))
        unsqueeze_name = self.next_name("unsqueeze")
        self.nodes.append(onnx.helper.make_node(
            "Unsqueeze",
            inputs=[self.oname, axes_name],
            outputs=[unsqueeze_name],
        ))
        self.oname = unsqueeze_name
        self.oshape = unsqueeze_shape(self.oshape, axes)
        return self

    def diagonalize(self, axis1: int, axis2: int):
        assert 0 <= axis1 < axis2 < len(self.oshape)
        dim = self.oshape[axis1]
        assert dim == self.oshape[axis2]
        if dim != 1:
            ndim = len(self.oshape)
            indices_shape = self.oshape[:axis1] + (1,) + self.oshape[axis1 + 1:]
            arange = np.arange(dim)
            expanded = np.expand_dims(arange, tuple(range(1, ndim - axis2)))
            indices = np.broadcast_to(expanded, indices_shape)
            indices_name = self.next_name("indices")
            self.nodes.append(make_constant_node(indices_name, indices))
            gather_name = self.next_name("gather")
            self.nodes.append(onnx.helper.make_node(
                "GatherElements",
                inputs=[self.oname, indices_name],
                outputs=[gather_name],
                axis=axis1,
            ))
            self.oname = gather_name
            self.oshape = indices_shape
        return self.squeeze([axis1])

    def reducesum(self, axes: List[int]):
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

    def transpose(self, perm: Tuple[int, ...]):
        assert sorted(perm) == list(range(len(perm)))
        assert len(perm) == len(self.oshape)
        if tuple(perm) == tuple(range(len(self.oshape))):
            return self
        transpose_name = self.next_name("transpose")
        self.nodes.append(onnx.helper.make_node(
            "Transpose",
            inputs=[self.oname],
            outputs=[transpose_name],
            perm=perm,
        ))
        self.oname = transpose_name
        self.oshape = transpose_seq(self.oshape, perm)
        return self

    def matmul(self, arg: Transform):
        if len(self.oshape) == 1 or len(arg.oshape) == 1:
            matmul_oshape = np.broadcast_shapes(self.oshape, arg.oshape)[:-1]
        else:
            assert np.broadcast_shapes(self.oshape[-1:], arg.oshape[-2:-1]), \
                "shouldn't raise ValueError: shape mismatch"
            matmul_oshape = \
                np.broadcast_shapes(self.oshape[:-2], arg.oshape[:-2]) \
                    + (self.oshape[-2], arg.oshape[-1])
        self.inputs.update(arg.inputs)
        self.nodes += arg.nodes
        matmul_name = self.next_name("matmul")
        self.nodes.append(onnx.helper.make_node(
            "MatMul",
            inputs=[self.oname, arg.oname],
            outputs=[matmul_name],
        ))
        self.oname = matmul_name
        self.oshape = matmul_oshape
        return self

    def mul(self, arg: Transform):
        mul_shape = np.broadcast_shapes(self.oshape, arg.oshape)
        self.inputs.update(arg.inputs)
        self.nodes += arg.nodes
        mul_name = self.next_name("mul")
        self.nodes.append(onnx.helper.make_node(
            "Mul",
            inputs=[self.oname, arg.oname],
            outputs=[mul_name],
        ))
        self.oname = mul_name
        self.oshape = mul_shape
        return self


def einsum_direct_model(equation, ishapes, dtype) -> onnx.ModelProto:
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
    ninputs = len(ishapes)
    assert ninputs <= 100 # for convenience to keep input names short
    in_name = lambda i: "in%02d" % i # sortable names for i < 100

    spec = einsum.einsum_spec(equation, ishapes)

    # In two cases the output is just zeros or empty:
    # (1) empty if there are any 0 dims in the output shape,
    # (2) zeros if there are any 0 dims in any input shape
    # (because they either occur in the output, which would be
    # empty, or will be eliminated by ReduceSum and become zeros).
    oshape = spec.output.shape
    if any(shape_size(shape) == 0 for shape in ishapes + [oshape]):
        tensor = np.zeros(oshape, dtype=dtype)
        out_name = "out"
        nodes = [make_constant_node(out_name, tensor)]
        inputs = [(in_name(i), ishapes[i]) for i in range(ninputs)]
        outputs = [(out_name, oshape)]
        return onnx.helper.make_model(
            make_typed_graph("einsum_constant", nodes, inputs, outputs, dtype)
        )

    # Each transform is either an onnx model transforming the input
    # at that position or just a string with the name of the input
    # which represents the identity transformation.
    transforms = [
        Transform(in_name(i), ishapes[i], dtype)
        for i in range(ninputs)
    ]

    for i in range(ninputs):
        # squeezing avoids broadcasting in contractions and it can potentially
        # be an optimization to skip some diagonalizations and reducesums and
        # axes to juggle in contractions
        spec, transforms = einsum_squeeze_input(spec, transforms, i)
        spec, transforms = einsum_diagonalize_input(spec, transforms, i)

    # einsum_squeeze_input() must be done on all inputs before
    # einsum_reducesum_input() on any input, because a squeeze of
    # a later input can enable reducesum of an earlier input
    for i in range(ninputs):
        spec, transforms = einsum_reducesum_input(spec, transforms, i)

    # TODO: optimize the contraction order
    while len(transforms) > 1:
        spec, transforms = einsum_contract_inputs(spec, transforms, 0, 1)

    transform = einsum_finalize(spec, transforms[0])
    return transform.model(f"einsum({equation})")

def einsum_squeeze_input(spec: einsum.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    axes = [a for a in range(len(shape)) if shape[a] == 1]
    transforms[i].squeeze(axes)
    for a in sorted(axes, reverse=True):
        del idxs[a]
        del shape[a]
    assert tuple(shape) == transforms[i].oshape
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_diagonalize_input(spec: einsum.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    for a in reversed(range(1, len(idxs))):
        idx = idxs[a]
        b = idxs.index(idx)
        if b != a:
            transforms[i].diagonalize(b, a)
            del idxs[b]
            del shape[b]
            assert tuple(shape) == transforms[i].oshape
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_reducesum_input(spec: einsum.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    assert len(idxs) == len(set(idxs)), \
        "duplicates indexes after diagonalization pass"
    idxs_in_other_inputs = {
        idx for ispec in omit(spec.inputs, i) for idx in ispec.idxs
    }
    idxs_only_in_i = set(idxs) - idxs_in_other_inputs - set(spec.output.idxs)
    axes = [idxs.index(idx) for idx in idxs_only_in_i]
    transforms[i].reducesum(axes)
    for a in sorted(axes, reverse=True):
        del idxs[a]
        del shape[a]
    assert tuple(shape) == transforms[i].oshape
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_transpose_input(spec: einsum.EinsumSpec, transforms: List[Transform], i, idxs_transposed):
    ispec = spec.inputs[i]
    perm = transpose_perm(ispec.idxs, idxs_transposed)
    transforms[i].transpose(perm)
    ispec.idxs = idxs_transposed
    ispec.shape = transpose_seq(ispec.shape, perm)
    assert transforms[i].oshape == ispec.shape
    return spec, transforms

def einsum_contract_inputs(spec: einsum.EinsumSpec, transforms: List[Transform], i, j):
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    ij_idxs = set(i_ispec.idxs) & set(j_ispec.idxs)
    idxs_in_other_inputs = {
        idx for ispec in omit(spec.inputs, i, j) for idx in ispec.idxs
    }
    idxs2keep = idxs_in_other_inputs.union(spec.output.idxs)
    idxs2reduce = ij_idxs.difference(idxs2keep)
    if len(idxs2reduce) == 0:
        return einsum_mul_inputs(spec, transforms, i, j)
    else:
        return einsum_matmul_inputs(spec, transforms, idxs2reduce, i, j)

# matmul is an optimization of mul followed by reducesum:
# def einsum_matmul_inputs(spec, transforms, idxs2reduce, i, j):
#   spec, transforms = einsum_mul_inputs(spec, transforms, i, j)
#   i -= j < i # j's removal may shift i one position to the left
#   return einsum_reducesum_input(spec, transforms, i)
#
def einsum_matmul_inputs(spec: einsum.EinsumSpec, transforms: List[Transform], idxs2reduce, i, j):
    # We assume that each of i and j have no repeated or reducible indexes.
    # (Any repeated or reducible indexes in each input were removed in
    # einsum_diagonalize_input() and einsum_reducesum_input() up front
    # and einsum_contract_inputs() doesn't produce any repeated or reducible
    # indexes.)
    #
    # Under this assumption the indexes in i and j fall in 4 buckets:
    # 1. Those that are reducible after or during contraction, namely those
    #    not in the output or any other remaining inputs. These appear in
    #    both i and j (as we assume no reducible indexes in each input).
    # 2. The other indexes that appear in both i and j,
    # 3. The indexes that appear in i and not in j.
    # 4. The indexes that appear in j and not in i.
    #
    # The indexes in the output of the contraction are the disjoint
    # union of buckets 2, 3, 4.

    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    i_idxs = i_ispec.idxs
    j_idxs = j_ispec.idxs
    ij_idxs = set(i_idxs) & set(j_idxs)
    ij_keep_idxs = [idx for idx in i_idxs if idx in ij_idxs - idxs2reduce]
    ij_reduce_idxs = [idx for idx in i_idxs if idx in idxs2reduce]
    i_idxs_unshared = [idx for idx in i_idxs if idx not in ij_idxs]
    j_idxs_unshared = [idx for idx in j_idxs if idx not in ij_idxs]

    i_idxs_transposed = ij_keep_idxs + i_idxs_unshared + ij_reduce_idxs
    spec, transforms = einsum_transpose_input(spec, transforms, i, i_idxs_transposed)
    j_idxs_transposed = ij_keep_idxs + ij_reduce_idxs + j_idxs_unshared
    spec, transforms = einsum_transpose_input(spec, transforms, j, j_idxs_transposed)
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]

    ij_keep_shape = j_ispec.shape[0:len(ij_keep_idxs)]
    ij_reduce_shape = j_ispec.shape[len(ij_keep_idxs):len(ij_idxs)]
    j_unshared_shape = j_ispec.shape[len(ij_idxs):]
    i_unshared_shape = i_ispec.shape[len(ij_keep_idxs):][:len(i_idxs_unshared)]
    ij_reduce_size = math.prod(ij_reduce_shape)
    i_unshared_size = math.prod(i_unshared_shape)
    j_unshared_size = math.prod(j_unshared_shape)

    transforms[i].reshape(ij_keep_shape + (i_unshared_size, ij_reduce_size))
    assert len(transforms[i].oshape) == len(ij_keep_idxs) + 2
    transforms[j].reshape(ij_keep_shape + (ij_reduce_size, j_unshared_size))
    assert len(transforms[j].oshape) == len(ij_keep_idxs) + 2
    transforms[i].matmul(transforms[j])
    assert transforms[i].oshape == ij_keep_shape + (i_unshared_size, j_unshared_size)
    final_shape = ij_keep_shape + i_unshared_shape + j_unshared_shape
    transforms[i].reshape(final_shape)
    i_ispec.shape = final_shape
    i_ispec.idxs = ij_keep_idxs + i_idxs_unshared + j_idxs_unshared
    assert len(i_ispec.idxs) == len(i_ispec.shape), f"{i_ispec}"
    del transforms[j]
    del spec.inputs[j]
    return spec, transforms

def einsum_mul_inputs(spec: einsum.EinsumSpec, transforms: List[Transform], i, j):
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    i_idxs = i_ispec.idxs
    j_idxs = j_ispec.idxs
    ij_idxs = set(i_idxs) & set(j_idxs)

    # transpose j so it ends with the idxs that also occur in i, in the same order
    j_idxs_unshared = [idx for idx in j_idxs if idx not in ij_idxs]
    j_idxs_shared = [idx for idx in i_idxs if idx in ij_idxs]
    j_idxs_transposed = j_idxs_unshared + j_idxs_shared
    spec, transforms = einsum_transpose_input(spec, transforms, j, j_idxs_transposed)
    j_ispec = spec.inputs[j]

    # unsqueeze j so ends with all i's idxs, in the same order
    axes = [a for a in range(-len(i_idxs), 0) if i_idxs[a] not in ij_idxs]
    j_idxs_unsqueezed = j_idxs_unshared + i_idxs
    transforms[j].unsqueeze(axes)
    j_ispec.idxs = j_idxs_unsqueezed
    j_ispec.shape = unsqueeze_shape(j_ispec.shape, axes)
    assert j_ispec.shape == transforms[j].oshape
    assert len(j_ispec.shape) == len(j_idxs_unsqueezed)

    # mul() broadcasts i to unsqueezed j's rank
    transforms[i].mul(transforms[j])
    i_ispec.idxs = j_idxs_unsqueezed
    i_ispec.shape = np.broadcast_shapes(i_ispec.shape, j_ispec.shape)
    assert i_ispec.shape == transforms[i].oshape
    assert len(i_ispec.shape) == len(j_idxs_unsqueezed)
    del transforms[j]
    del spec.inputs[j]
    return spec, transforms

def einsum_finalize(spec: einsum.EinsumSpec, transform: Transform):
    assert len(spec.inputs) == 1
    ispec = spec.inputs[0]
    in_idxs = set(ispec.idxs)
    out_idxs = spec.output.idxs
    assert in_idxs.issubset(set(out_idxs)), f"{in_idxs},{out_idxs}"
    assert all(idx in in_idxs or spec.idxs_map[idx] == 1 for idx in out_idxs)
    if einsum_is_identity_spec(spec):
        # The equation is the identity transformation.
        return transform
    squeezed_out_idxs = [idx for idx in out_idxs if idx in in_idxs]
    perm = tuple(ispec.idxs.index(idx) for idx in squeezed_out_idxs)
    transform.transpose(perm)
    axes = [a for a in range(len(out_idxs)) if out_idxs[a] not in in_idxs]
    return transform.unsqueeze(axes)


class Printer:
    verbosity: int
    counter: int
    printed_install_message: bool

    def __init__(self):
        self.verbosity = int(os.getenv("EINSUM_VERBOSE", "0"))
        self.counter = 0
        self.printed_install_message = False

    def print(self, equation, ishapes, model):
        if self.verbosity < 1:
            return
        name = equation.replace(',', '_comma_').replace('->', '_arrow_').replace(' ', '').replace('...', '_ellipses_')
        path = f"output/out%03d_{name}.py" % self.counter
        self.counter += 1
        print(equation, ishapes, path)
        try:
            import onnxconverter_common.onnx2py # type: ignore
            try:
                os.mkdir("output")
            except OSError:
                pass
            model_trace = onnxconverter_common.onnx2py.convert(model, path)
            py_obj = onnxconverter_common.onnx2py.TracingObject.get_py_obj(model_trace)
            print(repr(model_trace))
        except ImportError:
            if not self.printed_install_message:
                print("")
                print("Try `pip3 install onnxconverter-common` before running these tests to see the decomposed python code for each einsum equation")
                self.printed_install_message = True

def einsum_decomposed_model_test():
    print("einsum_decomposed_model_test() start")
    printer = Printer()

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
            # diagonalize axes s,t:
            ("ss->s", [(2,2)]),
            ("ssuu->su", [(2,2,3,3)]),
            ("sss->s", [(2,2,2)]),
            ("iss->is", [(3,2,2)]),
            ("sis->is", [(2,3,2)]),
            ("ssi->si", [(2,2,3)]),
            # reducesum axes s,t,u:
            ("sij->ij", [(4,2,3)]),
            ("isj->ij", [(2,4,3)]),
            ("ijs->ij", [(2,3,4)]),
            ("sitju->ij", [(4,2,5,3,6)]),
            # transpose:
            ("ij->ji", [(2,3)]),
            ("ijk->jik", [(2,3,4)]),
            ("ijk->jki", [(2,3,4)]),
            ("ijk->kji", [(2,3,4)]),
            ("ijk->ijk", [(2,3,4)]),
            ("ijk->ikj", [(2,3,4)]),
            ("ijk->kij", [(2,3,4)]),
            # unsqueeze:
            ("ij", [(1,2)]),
            ("ij->ji", [(1,2)]),
            ("ij", [(1,1)]),
            ("ij->ji", [(1,1)]),
            ("ghijk,ghjkm->ghim", [(1,5,2,1,3),(6,1,3,1,4)]),
            # matmul:
            ("ij,j", [(2,3),(3,)]),
            ("i,i", [(2,),(2,)]),
            ("ij,ij", [(2,3),(2,3)]),
            ("ij,ji", [(2,3),(3,2)]),
            ("ij,jk", [(2,3),(3,4)]),
            ("hij,hjk", [(5,2,3),(5,3,4)]),
            ("ghijk,ghjkm", [(6,5,2,3,3),(6,5,3,3,4)]),
            ("ghijk,ghjkm,gh", [(6,5,2,3,3),(6,5,3,3,4),(6,5)]),
            ("ghijk,ghjkm->ghim", [(6,5,2,3,3),(6,5,3,3,4)]),
        ]:
        inputs = [ np.random.rand(*shape) for shape in ishapes ]
        expected = np.einsum(equation, *inputs)
        model = einsum_decomposed_model(equation, ishapes, np.float64)
        printer.print(equation, ishapes, model)
        [actual] = infer_shapes_and_run_model(model, *inputs)
        assert expected.shape == actual.shape
        np.testing.assert_almost_equal(expected, actual)

    print("einsum_decomposed_model_test() end")

if __name__ == "__main__":
   einsum_direct_model_test()
   einsum_decomposed_model_test()

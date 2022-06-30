from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, KeysView, List, Optional, Sequence, Set, Tuple, TypeVar, Union
from copy import deepcopy
import math
import string
import itertools
import os
import numpy as np
import onnx # type: ignore
import onnxruntime # type: ignore

Shape = Sequence[int]
DType = Union[np.dtype, type]

# Shape helpers:

def nonneg(axes: Sequence[int], length: int, reverse=False, unique=True) -> Sequence[int]:
    assert all(-length <= a < length for a in axes)
    nn = sorted(((a + length) if a < 0 else a for a in axes), reverse=reverse)
    if unique: assert len(nn) == len(set(nn)), "duplicate axes"
    return nn

def shapeSize(shape: Shape) -> int: return math.prod(shape)

def shapeSplit(shape: Shape, *splits: int) -> Tuple[Shape, ...]:
    assert all(split >= 0 for split in splits)
    assert sum(splits) == len(shape)
    # TODO: come up with something simpler and faster than the following
    return tuple(shape[sum(splits[:i]):sum(splits[:i + 1])] for i in range(len(splits)))

def shapeConcat(*shape: Shape) -> Shape:
    return tuple(itertools.chain(*shape))

def shapeExpandDims(shape: Shape, axes: Sequence[int]) -> Shape:
    axes = nonneg(axes, len(shape) + len(axes), reverse=False)
    assert len(axes) == len(set(axes)), "duplicate axes"
    lst = list(shape)
    for a in axes:
        lst.insert(a, 1)
    return lst

X = TypeVar('X')
def listDeleteIdxs(lst: List[X], idxs: Sequence[int]) -> List[X]:
    idxs = nonneg(idxs, len(lst), reverse=True)
    assert len(idxs) == len(set(idxs)), "duplicate idxs"
    for i in idxs:
        del lst[i]
    return lst

Y = TypeVar('Y')
def seqTranspose(seq: Sequence[Y], perm: Sequence[int]) -> Sequence[Y]:
    return tuple(seq[a] for a in perm)

def transposePerm(original: Sequence[str], transposed: Sequence[str]) -> Sequence[int]:
    assert sorted(original) == sorted(transposed), f"'{original}', '{transposed}'"
    perm = tuple(original.index(x) for x in transposed)
    assert tuple(transposed) == seqTranspose(original, perm)
    return perm

# ONNX helpers:

def onnx_type(dtype : DType) -> onnx.TensorProto.DataType:
    '''Returns equivalent onnx.TensorProto.DataType for a given NumPy dtype
    where dtype can be either a np.dtype or np.float32, np.int64, etc.'''
    ty = np.dtype(dtype) # np.dtype() is idempotent
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty]

def param(param_name: str, dtype: DType, shape: Shape) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(
        param_name,
        onnx_type(dtype),
        shape)

def make_constant_node(output_name: str, tensor) -> onnx.NodeProto:
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
    graph = model.graph
    assert len(graph.input) == len(inputs)
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    inputs_dict = {gi.name: i for gi, i in zip(graph.input, inputs)}
    output_names = [go.name for go in graph.output]
    return sess.run(output_names, inputs_dict)

def infer_shapes_and_run_model(model: onnx.ModelProto, *inputs):
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return run_model(model, *inputs)


EINSUM_ELLIPSIS = "..."
EINSUM_ELLIPSIS_CHARS = string.digits
EINSUM_LETTERS = string.ascii_uppercase + string.ascii_lowercase # A-Za-z
EINSUM_LETTERS_SET = set(EINSUM_LETTERS)

@dataclass
class EinsumParam:
    name: str
    shape: Shape
    subscripts: str

    def __init__(self, name: str, shape: Shape, subscripts: str):
        self.name = name
        self.shape = shape
        # edit subscripts to make ellipsis dots match their shape
        front, ellipsis, tail = subscripts.partition(EINSUM_ELLIPSIS)
        if ellipsis:
            ellipsisLen = len(shape) - len(front) - len(tail)
            assert 0 <= ellipsisLen, \
                f"subscripts '{subscripts}' have more indices than rank of shape {shape}"
            assert ellipsisLen <= len(EINSUM_ELLIPSIS_CHARS), \
                f"ellipsis ran {ellipsisLen} exceeds maximum of {len(EINSUM_ELLIPSIS_CHARS)}"
            subscripts = front + EINSUM_ELLIPSIS_CHARS[:ellipsisLen] + tail
        assert len(subscripts) == len(shape)
        self.subscripts = subscripts

    def rank(self) -> int:
        return len(self.shape)

    def size(self) -> int:
        return math.prod(self.shape)

    def duplicates(self) -> Set[str]:
        return {s for s in self.subscripts if self.subscripts.count(s) > 1}

    def axes(self, subscriptsSubset: Set[str]) -> Sequence[int]:
        assert set(subscriptsSubset) <= set(self.subscripts)
        return [self.subscripts.index(s) for s in subscriptsSubset]

    def deleteAxes(self, axes: Sequence[int]) -> None:
        # self.shape = tuple(listDeleteIdxs(list(self.shape), axes))
        # self.subscripts = "".join(listDeleteIdxs(list(self.subscripts), axes))
        axes = nonneg(axes, len(self.shape), reverse=True)
        assert len(axes) == len(set(axes)), "duplicate axes"
        shape, subscripts = list(self.shape), list(self.subscripts)
        for a in axes:
            del shape[a]; del subscripts[a]
        self.shape, self.subscripts = tuple(shape), "".join(subscripts)

@dataclass
class Einsummer:
    dtype: DType
    inputs: List[EinsumParam]
    outputs: List[EinsumParam]
    result: EinsumParam # result.name is ignored
    nodes: List[onnx.NodeProto]

    def __init__(self, inputs: List[EinsumParam], result: EinsumParam, dtype: DType):
        self.dtype = dtype
        self.inputs = inputs
        # transform() mutates outputs, so we make a deep copy to avoud interference
        # with the inputs, which are needed to make a graph in the end
        self.outputs = deepcopy(inputs)
        self.result = result
        self.nodes = []

    def nextOutputName(self, prefix: str) -> str:
        return f"{prefix}_{len(self.nodes)}"

    def otherSubscripts(self, *ignore: EinsumParam) -> Set[str]:
        subscripts: Set[str] = set()
        for output in self.outputs:
            if output not in ignore:
                subscripts = subscripts.union(output.subscripts)
        if self.result not in ignore:
            subscripts = subscripts.union(self.result.subscripts)
        return subscripts

    def diagonalize(self, output: EinsumParam) -> None:
        assert output in self.outputs
        for subscript in output.duplicates():
            axes = [a for a, s in enumerate(output.subscripts) if s == subscript]
            self.diagonal(output, axes)
        assert not output.duplicates()

    def diagonal(self, output: EinsumParam, axes: Sequence[int]) -> None:
        subscript = output.subscripts[axes[0]]
        dim = output.shape[axes[0]]
        assert all(dim == output.shape[a] for a in axes)

        if dim == 1:
            self.squeeze(output, axes[1:])

        size = dim ** len(axes)
        mask = [False] * size
        assert (size - 1) % (dim - 1) == 0
        mask[0:size:(size - 1) // (dim - 1)] = [True] * dim
        assert np.array_equal(
            np.reshape(mask, (dim,)*len(axes)).nonzero(),
            (np.arange(dim),) * len(axes)), \
            "mask[0,...,0]==...==mask[dim-1,...,dim-1]==True"
        maskShape = tuple(dim if s == subscript else 1 for s in output.subscripts)
        maskTensor = np.reshape(mask, maskShape)
        maskName = self.nextOutputName("diag_mask")
        self.nodes.append(make_constant_node(maskName, maskTensor))

        zeroScalar = np.zeros((), self.dtype)
        zeroName = self.nextOutputName("diag_zero")
        self.nodes.append(make_constant_node(zeroName, zeroScalar))

        whereName = self.nextOutputName("diag_where")
        self.nodes.append(onnx.helper.make_node(
            "Where",
            inputs=[maskName, output.name, zeroName],
            outputs=[whereName],
        ))
        output.name = whereName

        self.sum(output, axes[1:])

    def reduce(self, output: EinsumParam) -> None:
        assert output in self.outputs
        assert not output.duplicates(), "duplicates should have been removed in diagonalize"
        keep = self.otherSubscripts(output)
        reducible = set(output.subscripts) - keep
        self.sum(output, output.axes(reducible))

    def sum(self, output: EinsumParam, axes: Sequence[int]) -> None:
        if not axes:
            return
        axesTensor = np.array(axes, dtype=np.int64)
        axesName = self.nextOutputName("sum_axes")
        self.nodes.append(make_constant_node(axesName, axesTensor))
        sumName = self.nextOutputName("sum")
        self.nodes.append(onnx.helper.make_node(
            "ReduceSum",
            inputs=[output.name, axesName],
            outputs=[sumName],
            keepdims=0,
        ))
        output.name = sumName
        output.deleteAxes(axes)

    def squeeze(self, output: EinsumParam, axes: Sequence[int]) -> None:
        # could be implemented with sum(output, axes)
        assert all(output.shape[a] == 1 for a in axes)
        if not axes:
            return
        axesTensor = np.array(axes, dtype=np.int64)
        axesName = self.nextOutputName("squeeze_axes")
        self.nodes.append(make_constant_node(axesName, axesTensor))
        squeezeName = self.nextOutputName("squeeze")
        self.nodes.append(onnx.helper.make_node(
            "Squeeze",
            inputs=[output.name, axesName],
            outputs=[squeezeName],
        ))
        output.name = squeezeName
        output.deleteAxes(axes)

    def unsqueeze(self, output: EinsumParam, unsqueezedSubscripts: str) -> None:
        assert set(output.subscripts) <= set(unsqueezedSubscripts)
        axes = [a for a, s in enumerate(unsqueezedSubscripts) if s not in output.subscripts]
        if not axes:
            return
        axesTensor = np.array(axes, dtype=np.int64)
        axesName = self.nextOutputName("unsqueeze_axes")
        self.nodes.append(make_constant_node(axesName, axesTensor))
        unsqueezeName = self.nextOutputName("unsqueeze")
        self.nodes.append(onnx.helper.make_node(
            "Unsqueeze",
            inputs=[output.name, axesName],
            outputs=[unsqueezeName],
        ))
        output.name = unsqueezeName
        output.subscripts = unsqueezedSubscripts
        output.shape = shapeExpandDims(output.shape, axes)

    def rename(self, output: EinsumParam, name:str) -> None:
        if name == output.name:
            return
        self.nodes.append(onnx.helper.make_node(
            "Identity",
            inputs=[output.name],
            outputs=[name],
        ))
        output.name = name

    def reshape(self, output: EinsumParam, reShape: Shape, reSubscripts: str) -> None:
        assert len(reShape) == len(reSubscripts)
        if output.shape == reShape:
            output.subscripts = reSubscripts
            return
        assert shapeSize(output.shape) == shapeSize(reShape) > 0
        shapeTensor = np.array(reShape, dtype=np.int64)
        shapeName = self.nextOutputName("reshape_shape")
        self.nodes.append(make_constant_node(shapeName, shapeTensor))
        reshapeName = self.nextOutputName("reshape")
        self.nodes.append(onnx.helper.make_node(
            "Reshape",
            inputs=[output.name, shapeName],
            outputs=[reshapeName],
            # allowzero attribute doesn't matter because our shapes never have zeros
        ))
        output.name = reshapeName
        output.shape = reShape
        output.subscripts = reSubscripts

    def transpose(self, output: EinsumParam, transposedSubscripts: str, name:str = None) -> None:
        if output.subscripts == transposedSubscripts:
            if name is not None:
                self.rename(output, name)
            return
        perm = transposePerm(output.subscripts, transposedSubscripts)
        if name is None:
            name = self.nextOutputName("transpose")
        self.nodes.append(onnx.helper.make_node(
            "Transpose",
            inputs=[output.name],
            outputs=[name],
            perm=perm,
        ))
        output.name = name
        output.shape = seqTranspose(output.shape, perm)
        output.subscripts = transposedSubscripts

    def contract(self, output1: EinsumParam, output2: EinsumParam) -> None:
        assert output1 in self.outputs
        assert output2 in self.outputs
        keep = self.otherSubscripts(output1, output2)
        intersection = set(output1.subscripts).intersection(output2.subscripts)
        reducible = intersection - keep
        if not reducible:
            self.mul(output1, output2)
        else:
            self.matmul(output1, output2, reducible)

    def mul(self, o1: EinsumParam, o2: EinsumParam) -> None:
        assert o1 in self.outputs
        assert o2 in self.outputs

        # transpose o1 to put the shared subscripts at the end, in the order they appear in o2
        shared = set(o1.subscripts).intersection(o2.subscripts)
        sharedSubscripts    = "".join(s for s in o2.subscripts if s in shared)
        subscripts1unshared = "".join(s for s in o1.subscripts if s not in shared)
        subscripts1transposed = subscripts1unshared + sharedSubscripts
        self.transpose(o1, subscripts1transposed)

        # unsqueeze to make o1 end in same subscripts as o2
        subscripts = subscripts1unshared + o2.subscripts
        self.unsqueeze(o1, subscripts)

        mulName = self.nextOutputName("mul")
        self.nodes.append(onnx.helper.make_node(
            "Mul",
            inputs=[o1.name, o2.name],
            outputs=[mulName],
        ))
        o1.name = mulName
        o1.subscripts = subscripts
        o1.shape = np.broadcast_shapes(o1.shape, o2.shape)
        self.outputs.remove(o2)

    def matmul(self, o1: EinsumParam, o2: EinsumParam, reducible: Set[str]) -> None:
        # could be implemented with self.mul(o1, o2); self.reduce(o1)
        assert o1 in self.outputs
        assert o2 in self.outputs
        assert reducible

        # transpose o1 to put the shared subscripts at the end, in the order they appear in o2
        shared = set(o1.subscripts).intersection(o2.subscripts)
        assert reducible <= shared
        sharedKeep = shared - reducible
        sharedKeepSubscripts = "".join(s for s in o1.subscripts if s in sharedKeep)
        reducibleSubscripts  = "".join(s for s in o1.subscripts if s in reducible)
        subscripts1unshared  = "".join(s for s in o1.subscripts if s not in shared)
        subscripts2unshared  = "".join(s for s in o2.subscripts if s not in shared)
        subscripts1transposed = sharedKeepSubscripts + subscripts1unshared + reducibleSubscripts
        subscripts2transposed = sharedKeepSubscripts + reducibleSubscripts + subscripts2unshared
        self.transpose(o1, subscripts1transposed)
        self.transpose(o2, subscripts2transposed)

        sharedKeep1Shape, unshared1Shape, reducibleShape = shapeSplit(o1.shape,
            len(sharedKeep), len(subscripts1unshared), len(reducible))
        sharedKeep2Shape, reducibleShape2, unshared2Shape = shapeSplit(o2.shape,
            len(sharedKeep), len(reducible), len(subscripts2unshared))
        assert reducibleShape == reducibleShape2, \
            "broadcast not needed because non-result 1-dim axes were squeezed at outset"

        # reshape unshared and redible dims into one dim each
        unshared1Size = shapeSize(unshared1Shape)
        reducibleSize = shapeSize(reducibleShape)
        unshared2Size = shapeSize(unshared2Shape)
        reShape1 = shapeConcat(sharedKeep1Shape, (unshared1Size, reducibleSize))
        reShape2 = shapeConcat(sharedKeep2Shape, (reducibleSize, unshared2Size))
        # "(", ")" are out-of-band subscripts for the reshaped unshared dims (which might
        # be empty); 1st reducible subscript is for reshaped reducible dims (non-empty)
        self.reshape(o1, reShape1, sharedKeepSubscripts + "(" + reducibleSubscripts[0])
        self.reshape(o2, reShape2, sharedKeepSubscripts + reducibleSubscripts[0] + ")")

        # matmul
        matmulName = self.nextOutputName("matmul")
        self.nodes.append(onnx.helper.make_node(
            "MatMul",
            inputs=[o1.name, o2.name],
            outputs=[matmulName],
        ))
        o1.name = matmulName
        o1.subscripts = sharedKeepSubscripts + "()"
        sharedKeepShape = np.broadcast_shapes(sharedKeep1Shape, sharedKeep2Shape)
        o1.shape = sharedKeepShape + (unshared1Size, unshared2Size)
        self.outputs.remove(o2)

        # reshape to get unshared dims back
        shape = shapeConcat(sharedKeepShape, unshared1Shape, unshared2Shape)
        subscripts = sharedKeepSubscripts + subscripts1unshared + subscripts2unshared
        self.reshape(o1, shape, subscripts)

    def finalize(self) -> None:
        assert len(self.outputs) == 1
        [output] = self.outputs
        self.transpose(output, self.result.subscripts, name=self.result.name)
        assert output == self.result, f"{output}, {self.result}"

    def transform(self) -> Einsummer:
        if self.result.size() == 0 or any(ou.size() == 0 for ou in self.outputs):
            # output is empty or all zeros (from ReduceSum of zero dim)
            self.nodes.append(make_constant_node(self.result.name, np.zeros(self.result.shape)))
            self.outputs = [self.result]
            return self
        for output in self.outputs:
            # Squeeze axes that don't appear in the result.
            # This avoids broadcast of reducible axes in matmul later on.
            # Must be run for all outputs before (diagonalize and) reduce pass
            # because it may enable more axes to reduce.
            nonresults = set(output.subscripts) - set(self.result.subscripts)
            self.squeeze(output, [a for a in output.axes(nonresults) if output.shape[a] == 1])
        for output in self.outputs:
            self.diagonalize(output)
            self.reduce(output)
        while len(self.outputs) > 1:
            self.contract(self.outputs[0], self.outputs[1])
        self.finalize()
        return self

    def graph(self, graph_name: str) -> onnx.GraphProto:
        assert self.outputs == [self.result], f"{self.outputs},\n{self.result}\nself:{self}"
        graph_inputs = [(i.name, i.shape) for i in self.inputs]
        graph_outputs = [(self.result.name, self.result.shape)]
        return make_typed_graph(graph_name, self.nodes, graph_inputs, graph_outputs, self.dtype)

    def model(self, graph_name: str) -> onnx.ModelProto:
        graph = self.graph(graph_name)
        return onnx.helper.make_model(graph)

def stripEllipsis(subscripts: str) -> str:
    return subscripts.replace(EINSUM_ELLIPSIS, "", 1)

def einsum_parse(equation: str) -> Tuple[List[str], str]:
    equation = equation.replace(" ", "")
    commaSeparatedArgs, arrow, output = equation.partition("->")
    args = commaSeparatedArgs.split(",")
    argsLetters = "".join(stripEllipsis(a) for a in args)
    assert set(argsLetters).issubset(string.ascii_letters)
    if arrow:
        stripped = stripEllipsis(output)
        assert len(stripped) == len(set(stripped)), "duplicate(s)"
        assert set(argsLetters).issuperset(stripped)
    else:
        output = EINSUM_ELLIPSIS + "".join(s for s in argsLetters if argsLetters.count(s) == 1)
    return args, output

def einsum_arg_dict(subscripts: str, shape: Shape) -> Tuple[Dict[str, int], Optional[Shape]]:
    # the following is copied from EinsumParam.__init__
    front, ellipsis, tail = subscripts.partition(EINSUM_ELLIPSIS)
    if ellipsis:
        ellipsisLen = len(shape) - len(front) - len(tail)
        assert 0 <= ellipsisLen, \
            f"subscripts '{subscripts}' have more indices than rank of shape {shape}"
        assert ellipsisLen <= len(EINSUM_ELLIPSIS_CHARS), \
            f"ellipsis ran {ellipsisLen} exceeds maximum of {len(EINSUM_ELLIPSIS_CHARS)}"
        ellipsisShape = shape[len(front):][:ellipsisLen]
    else:
        ellipsisShape = None
    letters = front + tail
    lettersShape = shapeConcat(shape[:len(front)], shape[len(shape) - len(tail):])
    dct = {s: d for s, d in zip(letters, lettersShape)}
    for s, d in zip(letters, lettersShape):
        assert dct[s] == d, "all occurrences of subscript in arg should have same dim"
    return dct, ellipsisShape

def einsum_infer_shape(args: Sequence[str], output: str, ishapes: Sequence[Shape]) -> Shape:
    assert len(args) == len(ishapes)
    dcts, ellipsisShapes = zip(*(einsum_arg_dict(arg, shape) for arg, shape in zip(args, ishapes)))
    ellipsisShapes = tuple(shape for shape in ellipsisShapes if shape != None)
    ellipsisLen = len(ellipsisShapes[0]) if ellipsisShapes else 0
    assert all(len(shape) == ellipsisLen for shape in ellipsisShapes)
    ellipsisShape: Shape = np.broadcast_shapes(*ellipsisShapes)
    front, ellipsis, tail = output.partition(EINSUM_ELLIPSIS)
    if not ellipsis: assert not ellipsisShape, \
            "ellipsis must be absent or empty in inputs when absent from output"
    def dim(letter: str) -> int:
        return np.broadcast_shapes(1, *(dct.get(letter, 1) for dct in dcts))[0]
    def shape(subscripts: str) -> Shape:
        return tuple(dim(s) for s in subscripts)
    return shapeConcat(shape(front), ellipsisShape, shape(tail))

def einsum_model(equation: str, ishapes: List[Shape], dtype: DType):
    args, output = einsum_parse(equation)
    shape = einsum_infer_shape(args, output, ishapes)
    inputs = [EinsumParam(f"in{i}", ishapes[i], args[i]) for i in range(len(ishapes))]
    result = EinsumParam("res", shape, output)
    return Einsummer(inputs, result, dtype).transform().model(f"einsum({equation})")

def einsum_run(equation: str, *tensors):
    model = einsum_model(equation, [t.shape for t in tensors], tensors[0].dtype)
    [result] = infer_shapes_and_run_model(model, *tensors)
    return result


VERBOSE = int(os.getenv("EINSUM_VERBOSE", "0")) > 0
def log(*args):
    if VERBOSE: print(*args)

def einsummer_basic_test():
    print("einsummer_basic_test() start")
    hline = "-" * 80
    log(hline)
    # zeros result because of zero dim in input
    log("zeroDimInInput:", Einsummer([
            EinsumParam("in", (0,2), "ij"),
        ], EinsumParam("res", (2,), "j"),
    np.uint32).transform())
    log(hline)
    # empty result because of zero dim in result
    log("zeroDimInResult:", Einsummer([
            EinsumParam("in", (0,2), "ij"),
        ], EinsumParam("res", (2,0), "ji"),
    float).transform())
    log(hline)
    # transpose
    log("transpose:", Einsummer([
            EinsumParam("in", (2,3), "ij"),
        ], EinsumParam("res", (3,2), "ji"),
    float).transform())
    log(hline)
    # ellipses
    log("ellipses:", Einsummer([
            EinsumParam("in1", (2,1,2,3,2), "a...ij"),
            EinsumParam("in2", (5,1,2,4,4), "...jkk"),
            EinsumParam("in3", (2,3,2,2,1,1,2,4,2), "xijx...xkx"),
        ], EinsumParam("res", (5,2,3,2), "...ij"),
    np.float32).transform())
    log(hline)
    print("einsummer_basic_test() end")

def einsum_model_test():
    print("einsum_model_test() start")

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
            # matmul with broadcast on reduction axes
            ("a,a", [(1,),(2,)]),
            ("ab,ab", [(1,3),(2,1)]),
            ("...ab,...ab", [(1,1,3),(4,2,1)]),
            ("abxy,abxz->xyz", [(1,3,4,5),(2,1,1,6)]),
        ]:
        inputs = [ np.random.rand(*shape) for shape in ishapes ]
        expected = np.einsum(equation, *inputs)
        model = einsum_model(equation, ishapes, np.float64)
        [actual] = infer_shapes_and_run_model(model, *inputs)
        assert expected.shape == actual.shape
        np.testing.assert_almost_equal(expected, actual, err_msg=f"{equation}, {ishapes}, {model}")

    print("einsum_model_test() end")

if __name__ == "__main__":
   einsummer_basic_test()
   einsum_model_test()

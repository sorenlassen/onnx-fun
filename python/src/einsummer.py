from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, KeysView, List, Sequence, Set, Tuple, TypeVar, Union
from copy import deepcopy
import math
import string
import os
import numpy as np
import onnx # type: ignore
import onnxruntime # type: ignore


Shape = Sequence[int]
DType = Union[np.dtype, type]


VERBOSE = int(os.getenv("EINSUM_VERBOSE", "0")) > 0
def log(*args):
    if VERBOSE: print(*args)


# Shape helpers:

def nonneg(axes: Sequence[int], length: int, reverse=False, unique=True) -> Sequence[int]:
    assert all(-length <= a < length for a in axes)
    nn = sorted(((a + length) if a < 0 else a for a in axes), reverse=reverse)
    if unique: assert len(nn) == len(set(nn)), "duplicate axes"
    return nn

def shapeSize(shape: Shape) -> int: return math.prod(shape)

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


# ONNX helpers:

def onnx_type(dtype : DType) -> onnx.TensorProto:
    '''Returns equivalent onnx.TensorProto basetype for a given NumPy type
    where dtype can be either a np.dtype or np.float32, np.int64, etc.'''
    ty = np.dtype(dtype) # np.dtype() is idempotent
    return {
        # TODO: support smaller int types (currently unsupported because of the type
        # constraints of ONNX ReduceSum and MatMul which we use to decompose Einsum)
        # TODO: support BFLOAT16 (currently unsupported by NumPy)
        np.dtype(np.int32): onnx.TensorProto.INT32,
        np.dtype(np.uint32): onnx.TensorProto.UINT32,
        np.dtype(np.int64): onnx.TensorProto.INT64,
        np.dtype(np.uint64): onnx.TensorProto.UINT64,
        np.dtype(np.float16): onnx.TensorProto.FLOAT16,
        np.dtype(np.float32): onnx.TensorProto.FLOAT,
        np.dtype(np.float64): onnx.TensorProto.DOUBLE,
        np.dtype(bool): onnx.TensorProto.BOOL,
    }[ty]

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


EINSUM_ELLIPSIS = "..."
EINSUM_ELLIPSIS_CHAR = "."
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
        # TODO: decide if caller should do this
        front, ellipsis, tail = subscripts.partition(EINSUM_ELLIPSIS)
        if ellipsis:
            ellipsisLen = len(shape) - len(front) - len(tail)
            assert ellipsisLen >= 0
            subscripts = front + EINSUM_ELLIPSIS_CHAR * ellipsisLen + tail
        assert len(subscripts) == len(shape)
        self.subscripts = subscripts

    def rank(self) -> int:
        return len(self.shape)

    def size(self) -> int:
        return math.prod(self.shape)

    def letters(self) -> Set[str]:
        return set(self.subscripts).difference(EINSUM_ELLIPSIS_CHAR)

    def duplicates(self) -> Set[str]:
        return {letter for letter in self.letters() if self.subscripts.count(letter) > 1}

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
        self.outputs = deepcopy(inputs)
        self.result = result
        self.nodes = []

    def nextOutputName(self, prefix: str) -> str:
        return f"{prefix}_{len(self.nodes)}"

    def occurs(self, letter: str, ignore: List[EinsumParam] = []) -> bool:
        for output in self.outputs:
            if output not in ignore and letter in output.subscripts:
                return True
        if self.result not in ignore:
            return letter in self.result.subscripts
        else:
            return False

    def diagonalize(self, output: EinsumParam) -> None:
        assert output in self.outputs
        for letter in output.duplicates():
            while output.subscripts.count(letter) > 1:
                axes = [a for a, s in enumerate(output.subscripts) if s == letter]
                log("diagonalize",self.outputs.index(output),letter,axes)
                self.diagonal(output, axes)
        assert not output.duplicates()

    def diagonal(self, output: EinsumParam, axes: Sequence[int]) -> None:
        letter = output.subscripts[axes[0]]
        dim = output.shape[axes[0]]
        assert all(dim == output.shape[a] for a in axes)

        if dim == 1:
            self.squeeze(output, axes[1:])

        size = dim ** len(axes)
        maskTensor = np.full(size, False)
        maskTensor[0:size:(size - 1) // (dim - 1)] = True
        assert np.array_equal(
            maskTensor.reshape((dim,)*len(axes)).nonzero(),
            (np.arange(dim),) * len(axes)), \
            "mask[0,...,0]==...==mask[dim-1,...,dim-1]==True"
        maskShape = tuple(dim if s == letter else 1 for s in output.subscripts)
        maskTensor = maskTensor.reshape(maskShape)
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

        self.reduceSum(output, axes[1:])

    def reduce(self, output: EinsumParam) -> None:
        assert output in self.outputs
        assert not output.duplicates(), "duplicates should have been removed in diagonalize"
        axes = [
            output.subscripts.index(letter)
            for letter in output.subscripts
            if not self.occurs(letter, ignore=[output])
        ]
        log("reduce",self.outputs.index(output),axes)
        self.reduceSum(output, axes)

    def reduceSum(self, output: EinsumParam, axes: Sequence[int]) -> None:
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
            keepdims=0,
        ))
        output.name = squeezeName
        output.deleteAxes(axes)

    def contract(self, output1: EinsumParam, output2: EinsumParam) -> None:
        # TODO
        self.outputs.remove(output2)

    def finalize(self) -> None:
        assert len(self.outputs) == 1
        [output] = self.outputs
        assert not output.duplicates()
        # TODO: uncomment the following assertion when preprocessing is complete
        # assert sorted(output.subscripts) == sorted(self.result.subscripts)
        if output.subscripts == self.result.subscripts:
            assert output.shape == self.result.shape
            if output.name != self.result.name:
                self.nodes.append(onnx.helper.make_node(
                    "Identity",
                    inputs=[output.name],
                    outputs=[self.result.name],
                ))
                output.name = self.result.name
            return
        # TODO: transpose
        ...

    def transform(self) -> Einsummer:
        if self.result.size() == 0 or any(ou.size() == 0 for ou in self.outputs):
            # output is empty or all zeros (from ReduceSum of zero dim)
            self.nodes.append(make_constant_node(self.result.name, np.zeros(self.result.shape)))
            return self
        for output in self.outputs:
            self.diagonalize(output)
            self.reduce(output)
        while len(self.outputs) > 1:
            self.contract(self.outputs[0], self.outputs[1])
        self.finalize()
        return self

def einsummer_test():
    print("einsummer_test() start")
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
    # ellipses
    log("ellipses:", Einsummer([
            EinsumParam("in1", (2,1,2,3,2), "a...ij"),
            EinsumParam("in2", (5,1,2,4,4), "...jkk"),
            EinsumParam("in3", (2,3,4,2,1,1,2,5,2), "xyzx...xwx"),
        ], EinsumParam("res", (3,4), "ik"),
    np.float32).transform())
    log(hline)
    print("einsummer_test() end")

if __name__ == "__main__":
   einsummer_test()

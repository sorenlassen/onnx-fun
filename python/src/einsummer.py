from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, KeysView, List, Sequence, Tuple, Union
from copy import deepcopy
import math
import string
import os
import numpy as np
import onnx # type: ignore
import onnxruntime # type: ignore


Shape = Tuple[int, ...]
DType = Union[np.dtype, type]


VERBOSE = int(os.getenv("EINSUM_VERBOSE", "0")) > 0
def log(*args):
    if VERBOSE: print(*args)


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
EINSUM_LETTERS = string.ascii_uppercase + string.ascii_lowercase # A-Za-z
EINSUM_LETTERS_SET = set(EINSUM_LETTERS)

@dataclass
class EinsumHistogram:
    histogram: Dict[str, int]

    def __init__(self, data: Sequence[str]):
        self.histogram = defaultdict(int)
        for datum in data:
            self.histogram[datum] += 1

    def keys(self) -> KeysView[str]:
        return self.histogram.keys()

    def __getitem__(self, datum: str) -> int:
        return self.histogram.get(datum, 0)

    def decrement(self, datum: str) -> None:
        c = self.histogram[datum]
        assert c > 0
        if c > 1:
            self.histogram[datum] = c - 1
        else:
            del self.histogram[datum]

@dataclass
class EinsumSubscripts:
    subscriptsList: List[str]
    ellipsisPos: int # len(subscripts) if no ellipsis
    ellipsisEnd: int # ellipsisPos if no ellipsis
    histogram: EinsumHistogram

    def __init__(self, subscriptsString: str):
        front, ellipsis, tail = subscriptsString.partition(EINSUM_ELLIPSIS)
        self.subscriptsList = list(front) + ([ellipsis] if ellipsis else []) + list(tail)
        letters = front + tail
        assert EINSUM_LETTERS_SET.issuperset(letters)
        self.ellipsisPos = len(front)
        self.ellipsisEnd = len(self.subscriptsList) - len(tail)
        assert (self.ellipsisEnd - self.ellipsisPos) == (1 if ellipsis else 0)
        self.histogram = EinsumHistogram(letters)

    def __delitem__(self, axis: int) -> None:
        assert (0 <= axis < self.ellipsisPos) or \
            (self.ellipsisEnd - len(self.subscriptsList) <= axis < 0)
        letter = self.subscriptsList[axis]
        self.histogram.decrement(letter)
        del self.subscriptsList[axis]
        if axis >= 0:
            self.ellipsisPos -= 1
            self.ellipsisEnd -= 1

    def index(self, letter: str, start: int = 0) -> int:
        i = self.subscriptsList.index(letter, start)
        if i < self.ellipsisPos:
            return i
        else:
            return i - len(self.subscriptsList)

@dataclass
class EinsumParam:
    name: str
    subscripts: EinsumSubscripts
    shape: Shape

    def rank(self) -> int:
        return len(self.shape)

    def size(self) -> int:
        return math.prod(self.shape)

    def delete(self, axis: int) -> None:
        assert -self.rank() <= axis < self.rank()
        del self.subscripts[axis]
        self.shape = self.shape[:axis] + self.shape[axis + 1:]

    def deleteAxes(self, axes: List[int]) -> None:
        offset = 0
        for a in sorted(axes):
            self.delete(a - offset)
            offset += a >= 0

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

    def occurs(self, letter: str, ignore: List[EinsumParam] = []) -> bool:
        for output in self.outputs:
            if output not in ignore and output.subscripts.histogram[letter] > 0:
                return True
        return self.result.subscripts.histogram[letter] > 0

    def diagonalize(self, output: EinsumParam) -> None:
        assert output in self.outputs
        for letter in output.subscripts.histogram.keys():
            while output.subscripts.histogram[letter] > 1:
                axis1 = output.subscripts.index(letter)
                axis2 = output.subscripts.index(letter, axis1 + 1)
                log("diagonalize",self.outputs.index(output),letter,axis1,axis2)
                # TODO: add nodes to diagonalize and set output.name to output of last node
                output.delete(axis1)

    def reduceSum(self, output: EinsumParam) -> None:
        assert output in self.outputs
        axes = [
            output.subscripts.index(letter)
            for letter in output.subscripts.histogram.keys()
            if not self.occurs(letter, ignore=[output])
        ]
        if axes:
            log("reduceSum",self.outputs.index(output),axes)
            # TODO: add node to ReduceSum and set output.name to node's output
            output.deleteAxes(axes)

    def contract(self, output1: EinsumParam, output2: EinsumParam) -> None:
        # TODO
        self.outputs.remove(output2)

    def finalize(self) -> None:
        # TODO
        ...

    def transform(self) -> None:
        if self.result.size() == 0 or any(ou.size() == 0 for ou in self.outputs):
            # output is empty or all zeros (from ReduceSum of zero dim)
            self.nodes.append(make_constant_node(self.result.name, np.zeros(self.result.shape)))
            return
        for output in self.outputs:
            self.diagonalize(output)
            self.reduceSum(output)
        while len(self.outputs) > 1:
            self.contract(self.outputs[0], self.outputs[1])
        self.finalize()

def einsummer_test():
    print("einsummer_test() start")

    in1_1 = EinsumParam("in1", EinsumSubscripts("a...ij"), (2,1,2,3,3,2))
    in1_2 = EinsumParam("in2", EinsumSubscripts("...jkk"), (5,1,3,2,4,4))
    in1_3 = EinsumParam("in3", EinsumSubscripts("xyzx...xwx"), (2,2,1,1,1,2,2))
    res1 = EinsumParam("res", EinsumSubscripts("ik"), (3,4))
    ein1 = Einsummer([in1_1, in1_2, in1_3], res1, np.float32)
    ein1.transform()
    log("ein1:",ein1)

    # zeros result because of zero dim in input
    in2 = EinsumParam("in", EinsumSubscripts("ij"), (0,2))
    res2 = EinsumParam("res", EinsumSubscripts("j"), (2,))
    ein2 = Einsummer([in2], res2, np.uint32)
    ein2.transform()
    log("ein2:",ein2)

    # empty result because of zero dim in result
    in3 = EinsumParam("in", EinsumSubscripts("ij"), (0,2))
    res3 = EinsumParam("res", EinsumSubscripts("ji"), (2,0))
    ein3 = Einsummer([in3], res3, float)
    ein3.transform()
    log("ein3:",ein3)

    print("einsummer_test() end")

if __name__ == "__main__":
   einsummer_test()

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, KeysView, List, Sequence, Tuple
from copy import deepcopy
import string
import os
import numpy as np


IntTuple = Tuple[int, ...]
Shape = IntTuple
DType = np.dtype
OnnxNode = Any # TODO: describe OnnxNode type better


VERBOSE = int(os.getenv("EINSUM_VERBOSE", "0")) > 0


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
        self.histogram = EinsumHistogram(list(letters))

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
    nodes: List[OnnxNode]

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
                if VERBOSE: print("diagonalize",self.outputs.index(output),letter,axis1,axis2)
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
            if VERBOSE: print("reduceSum",self.outputs.index(output),axes)
            # TODO: add node to ReduceSum and set output.name to node's output
            output.deleteAxes(axes)

    def contract(self, output1: EinsumParam, output2: EinsumParam) -> None:
        # TODO
        self.outputs.remove(output2)

    def finalize(self) -> None:
        # TODO
        ...

    def transform(self) -> None:
        for output in self.outputs:
            self.diagonalize(output)
            self.reduceSum(output)
        while len(self.outputs) > 1:
            self.contract(self.outputs[0], self.outputs[1])
        self.finalize()

def einsummer_test():
    print("einsummer_test() start")
    in1 = EinsumParam("in1", EinsumSubscripts("a...ij"), (2,1,2,3,3,2))
    in2 = EinsumParam("in2", EinsumSubscripts("...jkk"), (5,1,3,2,4,4))
    in3 = EinsumParam("in3", EinsumSubscripts("xyzx...xwx"), (2,2,1,1,1,2,2))
    res = EinsumParam("res", EinsumSubscripts("ik"), (3,4))
    ein = Einsummer([in1, in2, in3], res, DType(np.float32))
    ein.transform()
    if VERBOSE: print("ein:",ein)
    print("einsummer_test() end")

if __name__ == "__main__":
   einsummer_test()

# ONNX experiments

`einsum_decomposed_model(equation, ishapes, dtype)` in python/src/einsum_onnx.py creates an ONNX model for a given einsum equation and input shapes and type. Try it out as follows:
```python
python3 -i python/src/einsum_onnx.py # runs self test and takes you to python3 repl
>>> shapes,dtype=[(2,3,4),(2,4,5)],np.float32
>>> model=einsum_decomposed_model("bij,bjk",shapes,dtype)
>>> print(model) # displays all nodes, inputs, output
>>> [r234,r245]=[np.random.rand(*s).astype(dtype) for s in shapes]
>>> [result]=run_model(model,r234,r245)
>>> np.allclose(result,np.einsum("bij,bjk",r234,r245) # prints True
```

Silence the verbose test output by setting environment varable `EINSUM_VERBOSE=0`.

Type check with:
```bash
python3 -m mypy python/src/einsum_onnx.py
```

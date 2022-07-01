import os

class EinsumONNXModelPrinter:
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

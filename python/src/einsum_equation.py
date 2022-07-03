import re

P=r"[ a-z]*(?:\. *\. *\.[ a-z]*)?"
EINSUM_EQUATION_PATTERN = re.compile(rf"(^{P}(?:,{P})*)(?:- *>({P}))?$", re.A | re.I)

def einsum_equation_is_valid(equation: str) -> bool:
    match = EINSUM_EQUATION_PATTERN.match(equation)
    if match is None:
        return False # doesn't match regular expression
    output = match.group(2)
    if output is None:
        return True
    subscripts = output
    inputs = match.group(1)
    stripped = output.replace(" ", "").replace(".", "")
    strippedSet = set(stripped)
    if len(stripped) > len(strippedSet):
        return False # repeated subscripts in output
    if not strippedSet.issubset(inputs):
        return False # output contains subscript not in inputs
    return True

def einsum_equation_infer_output(equation: str) -> str:
    counts = {s: equation.count(s) for s in set(equation).difference(",. ")}
    output = "..." + "".join(s for s in sorted(counts) if counts[s] == 1)
    return equation + "->" + output

def einsum_equation_test():
    print("einsum_equation_test() start")

    for equation in [
        "ii->i",
    ]:
        assert einsum_equation_is_valid(equation), f"'{equation}'"

    for equation in [
        "ii->ij",
        "ii->ii",
    ]:
        assert not einsum_equation_is_valid(equation), f"'{equation}'"

    for equation, inferred in [
        ("...", "..."),
        ("Ab", "...Ab"),
        ("aB", "...Ba"),
        ("ba", "...ab"),
    ]:
        assert einsum_equation_is_valid(equation), f"'{equation}'"
        assert einsum_equation_infer_output(equation) == equation + "->" + inferred, \
            f"'{equation}' '{inferred}' '{einsum_equation_infer_output(equation)}'"

    print("einsum_equation_test() end")

if __name__ == "__main__":
   einsum_equation_test()

"""Choose the packed CKKS layout and lower a parsed public affine transform."""
from pathlib import Path

from crossfhe.affine import import_affine_onnx
from crossfhe.frontend import FrontendError


def import_packed_onnx(path: str | Path, *, max_elements=50000):
    matrix, bias, manifest = import_affine_onnx(path, max_elements=max_elements)
    rows, cols = matrix.shape
    period = 1 << (max(rows, cols)-1).bit_length()
    if period > 16384: raise FrontendError("packed period exceeds the supported 16384 slots")

    def dense(value):
        # Hexadecimal MLIR f64 literals preserve the exact public coefficients.
        import struct
        flat = ["0x" + struct.pack(">d", float(x)).hex().upper() for x in value.flat]
        if value.ndim == 2:
            data = "[" + ", ".join("[" + ", ".join(flat[i:i+cols]) + "]" for i in range(0, len(flat), cols)) + "]"
        else:
            data = "[" + ", ".join(flat) + "]"
        return f"dense<{data}> : tensor<{'x'.join(map(str, value.shape))}xf64>"

    cipher = f"!heir.rlwecipher<{period} x f64>"
    mlir = (f"module {{\n  func.func @model(%x: {cipher}) -> {cipher} "
            f"attributes {{ppml.fhe_enabled = true}} {{\n"
            f'    %y = "heir.matvec"(%x) {{matrix = {dense(matrix)}, bias = {dense(bias)}}} : ({cipher}) -> {cipher}\n'
            f"    return %y : {cipher}\n  }}\n}}\n")
    # Preserve the original packed import's field order as well as its values.
    return mlir, {
        "inputs": manifest["inputs"], "outputs": manifest["outputs"],
        "logic_comparisons": manifest["logic_comparisons"], "scalar_operations": manifest["scalar_operations"],
        "numeric_semantics": "public affine coefficients promoted to CKKS f64",
        "model_parameters": manifest["model_parameters"], "layout": "periodic-packed", "packed_period": period,
    }

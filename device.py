"""ONNX Runtime device selection (shared by OCR and LaMa)."""
import sys


def onnx_providers(device):
    """'auto' / 'cuda' / 'dml' / 'cpu' -> (resolved device, onnxruntime providers)"""
    import onnxruntime as ort
    try:
        ort.preload_dlls()  # use the pip-installed CUDA/cuDNN (onnxruntime-gpu[cuda,cudnn])
    except Exception:
        pass
    avail = ort.get_available_providers()
    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in avail else (
            "dml" if "DmlExecutionProvider" in avail else "cpu")
    prov = {"cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "cpu": ["CPUExecutionProvider"]}[device]
    if prov[0] not in avail:
        sys.exit(f"Device {device} is not available; onnxruntime providers: {avail}")
    return device, prov

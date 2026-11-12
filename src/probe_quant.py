import torch
import torch.nn as nn

print("torch:", torch.__version__)
print("supported engines:", torch.backends.quantized.supported_engines)
print("engine before:", torch.backends.quantized.engine)

if torch.backends.quantized.engine == "none":
    for e in ("fbgemm", "qnnpack"):
        if e in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = e
            break
print("engine after:", torch.backends.quantized.engine)

try:
    import torch.ao.quantization as q
    print("quantize_dynamic exists:", hasattr(q, "quantize_dynamic"))
except Exception as e:
    print("torch.ao.quantization import failed:", type(e).__name__, e)

try:
    import torchao
    print("torchao:", torchao.__version__)
except Exception:
    print("torchao: not installed")

toy = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).eval()
x = torch.randn(4, 64)
try:
    from torch.ao.quantization import quantize_dynamic
    qm = quantize_dynamic(toy, {nn.Linear}, dtype=torch.qint8)
    print("toy quantize OK | max output diff:", (toy(x) - qm(x)).abs().max().item())
    print(qm)
except Exception as e:
    print("toy quantize FAILED:", type(e).__name__, e)
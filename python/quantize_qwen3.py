import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "qwen3-embedding-0.6b.onnx"
OUTPUT_MODEL = "qwen3-embedding-0.6b-int8.onnx"

print("Loading model...")
model = onnx.load(INPUT_MODEL)

print("Starting dynamic quantization...")

quantize_dynamic(
    model_input=INPUT_MODEL,
    model_output=OUTPUT_MODEL,
    weight_type=QuantType.QInt8#,   # INT8 weights
    #optimize_model=True
)

print("Quantization complete.")
print("Saved to:", OUTPUT_MODEL)

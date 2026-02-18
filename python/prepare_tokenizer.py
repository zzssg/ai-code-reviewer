from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("onnx-community/Qwen3-Embedding-0.6B-ONNX")
tokenizer.save_pretrained("./models_Qwen3-Embedding-0.6B-ONNX")
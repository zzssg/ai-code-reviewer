import torch
from transformers import AutoModel

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Load model
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.eval()
model.config.use_cache = False

# Dummy inputs
input_ids = torch.ones(1, 16, dtype=torch.long)
attention_mask = torch.ones(1, 16, dtype=torch.long)

# Wrap forward to return only last_hidden_state
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        return outputs.last_hidden_state

wrapped_model = Wrapper(model)

# Export
torch.onnx.export(
    wrapped_model,
    (input_ids, attention_mask),
    "qwen3-embedding-0.6b.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "last_hidden_state": {1: "sequence"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print("Export complete.")

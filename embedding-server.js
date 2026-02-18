import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MODELS_DIR = path.join(__dirname, "models");

const app = express();
app.use(express.json({ limit: "10mb" }));

let tokenizer;
let session;

// -----------------------------
// CONFIG
// -----------------------------
const MAX_LENGTH = 256;

// -----------------------------
// TOKENIZER LOADING
// -----------------------------
async function initTokenizer() {
  const tokenizerJsonPath = path.join(MODELS_DIR, "tokenizer_q3-emb-0.6b.json");
  const tokenizerConfigPath = path.join(MODELS_DIR, "tokenizer_config_q3-emb-0.6b.json");

  const tokenizerJson = JSON.parse(fs.readFileSync(tokenizerJsonPath, "utf-8"));
  const tokenizerConfig = JSON.parse(fs.readFileSync(tokenizerConfigPath, "utf-8"));

  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  console.log("Tokenizer loaded successfully");
}

// -----------------------------
// MODEL INIT (MEMORY OPTIMIZED)
// -----------------------------
async function initModel() {
  const modelPath = path.join(MODELS_DIR, "qwen3-embedding-0.6b-int8.onnx");

  // For the options available see https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    executionMode: "sequential",
    intraOpNumThreads: 1,
    interOpNumThreads: 1
  });

  console.log("ONNX model loaded");
  console.log("Model inputs:", session.inputNames);
}

// -----------------------------
// TOKENIZATION
// -----------------------------
function tokenize(text) {
  const encoding = tokenizer.encode(text);

  let inputIds = encoding.ids.slice(0, MAX_LENGTH);
  let attentionMask = new Array(inputIds.length).fill(1);

  return { inputIds, attentionMask };
}

// -----------------------------
// L2 NORMALIZATION
// -----------------------------
function l2Normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    sum += vector[i] * vector[i];
  }

  const norm = Math.sqrt(sum) || 1;

  for (let i = 0; i < vector.length; i++) {
    vector[i] /= norm;
  }

  return vector;
}

// -----------------------------
// MEAN POOLING
// -----------------------------
function meanPool(hiddenStates, attentionMask) {
  const seqLength = attentionMask.length;
  const hiddenSize = hiddenStates.length / seqLength;

  const pooled = new Float32Array(hiddenSize);
  let validTokens = 0;

  for (let i = 0; i < seqLength; i++) {
    if (attentionMask[i] === 0) continue;
    validTokens++;

    for (let j = 0; j < hiddenSize; j++) {
      pooled[j] += hiddenStates[i * hiddenSize + j];
    }
  }

  if (validTokens > 0) {
    for (let j = 0; j < hiddenSize; j++) {
      pooled[j] /= validTokens;
    }
  }

  return l2Normalize(pooled);
}

// -----------------------------
// EMBEDDING
// -----------------------------
async function embed(text) {
  const { inputIds, attentionMask } = tokenize(text);
  const seqLength = inputIds.length;

  const inputIdsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(inputIds.map(BigInt)),
    [1, seqLength]
  );

  const attentionMaskTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(attentionMask.map(BigInt)),
    [1, seqLength]
  );

  const positionIdsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(
      Array.from({ length: seqLength }, (_, i) => BigInt(i))
    ),
    [1, seqLength]
  );

  const outputs = await session.run({
    input_ids: inputIdsTensor,
    attention_mask: attentionMaskTensor,
    position_ids: positionIdsTensor
  });

  const hidden = outputs.last_hidden_state.data;
  return Array.from(meanPool(hidden, attentionMask));
}

// -----------------------------
// API ENDPOINT
// -----------------------------
app.post("/api/embedding", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing text" });

    const embedding = await embed(text);

    res.json({
      dimensions: embedding.length,
      embedding
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// -----------------------------
// START SERVER (SINGLE WORKER)
// -----------------------------
(async () => {
  await initTokenizer();
  await initModel();

  app.listen(PORT, () =>
    console.log(`Embedding server running on port ${PORT}`)
  );
})();

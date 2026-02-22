import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import cluster from "cluster";
import ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MODELS_DIR = path.join(__dirname, "models");
const MODEL_NAME = process.env.EMB_MODEL_NAME || "qwen3-embedding-0.6b";

const app = express();
app.use(express.json({ limit: "10mb" }));

let tokenizerHF;
let session;
let modelInputs;

// -----------------------------
// CONFIG
// -----------------------------
const MAX_LENGTH = 256;

// -----------------------------
// TOKENIZER LOADING
// -----------------------------
async function initTokenizer() {
  const tokenizerPath = path.join(MODELS_DIR, `tokenizer_${MODEL_NAME}.json`);
  const tokenizerConfigPath = path.join(MODELS_DIR, `tokenizer_config_${MODEL_NAME}.json`);

  const tokenizer = JSON.parse(fs.readFileSync(tokenizerPath, "utf-8"));
  const tokenizerConfig = JSON.parse(fs.readFileSync(tokenizerConfigPath, "utf-8"));

  tokenizerHF = new Tokenizer(tokenizer, tokenizerConfig);
  console.log(`Tokenizer loaded successfully using paths: ${tokenizerPath}, ${tokenizerConfigPath}`);
}

// -----------------------------
// MODEL INIT (MEMORY OPTIMIZED)
// -----------------------------
async function initModel() {
  const modelPath = path.join(MODELS_DIR, `${MODEL_NAME}.onnx`);

  // For the options available see https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    executionMode: "parallel",
    graphOptimizationLevel: "all"
  });

  console.log(`ONNX model '${MODEL_NAME}' loaded using path: ${modelPath}`);
  modelInputs = session.inputNames;
  console.log(`Model inputs: ${modelInputs.join(", ")}`);
}

// -----------------------------
// TOKENIZATION
// -----------------------------
function tokenizeLongText(text, maxLength = MAX_LENGTH, overlap = 64) {
  const encoding = tokenizerHF.encode(text);
  const inputIds = encoding.ids;

  if (inputIds.length <= maxLength - 2) {
    return [{ inputIds: inputIds.slice(0, maxLength), attentionMask: new Array(inputIds.length).fill(1) }];
  }
  const chunks = [];
  for (let start = 0; start < inputIds.length; start += maxLength - overlap) {
    const end = Math.min(start + maxLength - 2, inputIds.length);
    chunks.push({
      inputIds: inputIds.slice(start, end),
      attentionMask: new Array(end - start).fill(1)
    });
  }
  return chunks;
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
async function embedChunk({ inputIds, attentionMask }) {
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

  let runInputs = {
    input_ids: inputIdsTensor,
    attention_mask: attentionMaskTensor
  };
  if (modelInputs.includes("position_ids")) {
    runInputs.position_ids = positionIdsTensor;
  }

  const outputs = await session.run(runInputs);

  const hidden = outputs.last_hidden_state.data;
  return Array.from(meanPool(hidden, attentionMask));
}

// -----------------------------
// API ENDPOINT
// -----------------------------
app.post("/v1/embeddings", async (req, res) => {
  try {
    const { input } = req.body;
    if (!input) return res.status(400).json({ error: "Missing input" });

    const chunks = tokenizeLongText(input);
    const embeddings = [];
    for (const chunk of chunks) {
      const emb = await embedChunk(chunk);
      embeddings.push(emb);
    }

    const data = embeddings.map((emb, idx) => {
      return {
        object: "embedding",
        embedding: Array.isArray(emb) ? emb : [],
        index: idx,
      }
    });

    res.json({ 
      object: "list", 
      data, 
      model: MODEL_NAME,
      "usage": {
        "prompt_tokens": input.length, 
        "total_tokens": 0
      } 
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// -----------------------------
// START CLUSTER SERVER
// -----------------------------
if (cluster.isPrimary) {
  const workers = process.env.EMB_WORKERS ? parseInt(process.env.EMB_WORKERS) : 2;
  console.log(`Primary process is running. Forking ${workers} workers...`);
  for (let i = 0; i < workers; i++) {
    cluster.fork();
  }
  cluster.on("exit", (worker, code, signal) => {
    console.log(`Worker ${worker.process.pid} died. Restarting...`);
    cluster.fork();
  });
} else {
  (async () => {
  await initTokenizer();
  await initModel();

  app.listen(PORT, () =>
    console.log(`Embedding server running on port ${PORT}`)
  );
})();
}

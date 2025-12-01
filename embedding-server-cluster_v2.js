// embedding-server-cluster.js (offline, no @xenova/tokenizers)
import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { cpus } from "os";
import cluster from "cluster";
import ort from "onnxruntime-node";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MODELS_DIR = path.join(__dirname, "models");

const app = express();
app.use(express.json({ limit: "10mb" }));

// --- Load vocab.json ---
const vocabPath = path.join(MODELS_DIR, "all-MiniLM-L6-v2-vocab.json");
const vocabRaw = JSON.parse(fs.readFileSync(vocabPath, "utf-8"));

// Ensure all vocab entries are numbers
const vocab = {};
for (const [token, id] of Object.entries(vocabRaw)) {
  vocab[token] = typeof id === "number" ? id : 0;
}

const CLS_TOKEN = "[CLS]";
const SEP_TOKEN = "[SEP]";
const PAD_TOKEN = "[PAD]";
const UNK_TOKEN = "[UNK]";

// --- Minimal tokenizer ---
function tokenize(text, maxLength = 128) {
  const tokens = text.trim().split(/\s+/).slice(0, maxLength - 2);
  const inputTokens = [CLS_TOKEN, ...tokens, SEP_TOKEN];

  const inputIds = inputTokens.map(t => vocab[t] ?? vocab[UNK_TOKEN] ?? 0);
  const attentionMask = inputIds.map(id => (id === (vocab[PAD_TOKEN] ?? 0) ? 0 : 1));

  return { inputIds, attentionMask };
}

// --- Load ONNX model ---
const modelPath = path.join(MODELS_DIR, "all-MiniLM-L6-v2.onnx");
let session;

async function initModel() {
  session = await ort.InferenceSession.create(modelPath);
  console.log("ONNX model loaded successfully");
}

// --- Mean pooling ---
function meanPool(hiddenStates, attentionMask) {
  const seqLength = attentionMask.length;
  const hiddenSize = hiddenStates.length / seqLength;
  const pooled = new Array(hiddenSize).fill(0);
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
  return pooled;
}

// --- Generate embedding ---
async function getEmbedding(text) {
  const { inputIds, attentionMask } = tokenize(text);

  // Create tensors
  const inputIdsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(inputIds.map(BigInt)),
    [1, inputIds.length]
  );

  const attentionMaskTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(attentionMask.map(BigInt)),
    [1, attentionMask.length]
  );

  const tokenTypeIdsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(new Array(inputIds.length).fill(0).map(BigInt)),
    [1, inputIds.length]
  );

  const feeds = {
    input_ids: inputIdsTensor,
    attention_mask: attentionMaskTensor,
    token_type_ids: tokenTypeIdsTensor,
  };

  const results = await session.run(feeds);
  const hiddenStates = results["last_hidden_state"].data;
  const embedding = meanPool(hiddenStates, attentionMask);
  return embedding;
}

// --- API endpoint ---
app.post("/api/embedding", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing text" });

    const embedding = await getEmbedding(text);
    res.json({ embedding });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// --- Cluster mode ---
if (cluster.isPrimary) {
  const numWorkers = cpus().length;
  console.log(`Master cluster setting up ${numWorkers} workers...`);

  for (let i = 0; i < numWorkers; i++) cluster.fork();

  cluster.on("exit", (worker) => {
    console.log(`Worker ${worker.process.pid} died. Spawning a new one.`);
    cluster.fork();
  });
} else {
  initModel().then(() => {
    app.listen(PORT, () => {
      console.log(`Worker ${process.pid} running on port ${PORT}`);
    });
  });
}

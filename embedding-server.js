import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { cpus } from "os";
import fs from "fs";
import cluster from "cluster";
import ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const MODELS_DIR = path.join(__dirname, "models");

const app = express();
app.use(express.json({ limit: "10mb" }));

let tokenizer;

// -----------------------------
// TOKENIZER LOADING
// -----------------------------
async function initTokenizer() {
  const tokenizerJsonPath = path.join(MODELS_DIR, "tokenizer.json");
  const tokenizerConfigPath = path.join(MODELS_DIR, "tokenizer_config.json");

  const tokenizerJson = JSON.parse(fs.readFileSync(tokenizerJsonPath, "utf-8"));
  const tokenizerConfig = JSON.parse(fs.readFileSync(tokenizerConfigPath, "utf-8"));

  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  console.log("Tokenizer loaded successfully");
}

// -----------------------------
// TOKENIZATION & CHUNKING
// -----------------------------
const CLS_ID = 101; // default [CLS] ID for many HF models, adjust if needed
const SEP_ID = 102; // default [SEP] ID
const PAD_ID = 0;   // default [PAD] ID

function tokenizeChunk(tokenIds) {
  const inputIds = [CLS_ID, ...tokenIds, SEP_ID];
  const attentionMask = inputIds.map(id => (id === PAD_ID ? 0 : 1));
  return { inputIds, attentionMask };
}

function tokenizeLongText(text, maxLength = 256, overlap = 64) {
  if (!tokenizer) throw new Error("Tokenizer not initialized");

  const encoding = tokenizer.encode(text);
  const tokenIds = encoding.ids;

  if (tokenIds.length <= maxLength - 2) {
    return [tokenizeChunk(tokenIds)];
  }

  const chunks = [];
  let start = 0;
  while (start < tokenIds.length) {
    const end = start + (maxLength - 2);
    const chunkIds = tokenIds.slice(start, end);
    chunks.push(tokenizeChunk(chunkIds));
    start += (maxLength - 2) - overlap;
  }

  return chunks;
}

// -----------------------------
// MODEL LOADING
// -----------------------------
const modelPath = path.join(MODELS_DIR, "all-MiniLM-L6-v2.onnx");
let session;

async function initModel() {
  try {
    session = await ort.InferenceSession.create(modelPath);
    console.log("ONNX model loaded successfully");
  } catch (e) {
    console.error("Failed to load ONNX model at path:", modelPath, e);
    process.exit(1);
  }
}

// -----------------------------
// MEAN POOLING
// -----------------------------
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

// -----------------------------
// EMBEDDING A SINGLE CHUNK
// -----------------------------
async function embedChunk({ inputIds, attentionMask }) {
  const seqLength = inputIds.length;

  const inputIdsTensor = new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, seqLength]);
  const attentionMaskTensor = new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, seqLength]);
  const tokenTypeIdsTensor = new ort.Tensor("int64", BigInt64Array.from(new Array(seqLength).fill(0).map(BigInt)), [1, seqLength]);

  const outputs = await session.run({
    input_ids: inputIdsTensor,
    attention_mask: attentionMaskTensor,
    token_type_ids: tokenTypeIdsTensor
  });

  return meanPool(outputs["last_hidden_state"].data, attentionMask);
}

// -----------------------------
// API ENDPOINT
// -----------------------------
app.post("/api/embedding", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing text" });

    const chunks = tokenizeLongText(text);
    const embeddings = [];

    for (const chunk of chunks) {
      const emb = await embedChunk(chunk);
      embeddings.push(emb);
    }

    res.json({ chunks: chunks.length, embeddings });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.toString() });
  }
});

// -----------------------------
// CLUSTER SERVER
// -----------------------------
if (cluster.isPrimary) {
  const workers = process.env.EMB_WORKERS || Math.max(4, Math.floor(cpus().length / 2));
  console.log(`Master starting ${workers} workers...`);
  for (let i = 0; i < workers; i++) cluster.fork();

  cluster.on("exit", worker => {
    console.log(`Worker ${worker.process.pid} died. Restarting...`);
    cluster.fork();
  });
} else {
  (async () => {
    await initTokenizer();
    await initModel();
    app.listen(PORT, () => console.log(`Worker ${process.pid} listening on port ${PORT}`));
  })();
}

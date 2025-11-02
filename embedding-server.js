import express from "express";
import { pipeline, env } from "@xenova/transformers";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODELS_DIR = path.join(__dirname, "models");

// Disable all remote downloads
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = MODELS_DIR;

const app = express();
app.use(express.json());

let embedder = null;

// Lazy-load model
async function getEmbedder() {
  if (!embedder) {
    console.log(`Loading local embedding model from MODELS_DIR: ${MODELS_DIR}`);
    embedder = await pipeline(
      "feature-extraction",
      path.join(".", "all-MiniLM-L6-v2")
    );
    console.log("Model loaded from", MODELS_DIR);
  }
  return embedder;
}

app.post("/api/embedding", async (req, res) => {
  let eb_duration = Date.now();
  try {
    const text = req.body.text;
    if (!text) {
      return res.status(400).json({ error: "Missing 'text' field in request body" });
    }

    const embedder = await getEmbedder();
    const result = await embedder(text, { pooling: "mean", normalize: true });
    const embedding = Array.from(result.data);

    res.json({ embedding });
	eb_duration = Date.now() - eb_duration;
	console.log(`Embeeding produced in ${eb_duration} ms`);
  } catch (err) {
    console.error("Embedding error:", err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () =>
  console.log(`Local embedding server running on http://localhost:${PORT}`)
);

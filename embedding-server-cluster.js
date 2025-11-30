import cluster from "cluster";
import os from "os";
import express from "express";
import { pipeline, env } from "@xenova/transformers";
import path from "path";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// LOGGER SETUP
// ---------------------------------------------------------------------------
import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODELS_DIR = path.join(__dirname, "models");
const MODEL_NAME = "all-MiniLM-L6-v2";

const PORT = process.env.PORT || 3000;
const WORKERS = process.env.WORKERS || Math.max(1, os.cpus().length - 1);

// Disable remote downloads
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = MODELS_DIR;

// ---------------------------------------------------------------------------
// MASTER PROCESS
// ---------------------------------------------------------------------------
if (cluster.isPrimary) {
  log.info(`Master ${process.pid} starting ${WORKERS} workers...`);

  for (let i = 0; i < WORKERS; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker) => {
    log.info(`Worker ${worker.process.pid} died. Restarting...`);
    cluster.fork();
  });

  log.info(`Embedding server running with ${WORKERS} workers on port ${PORT}`);
}

// ---------------------------------------------------------------------------
// WORKER PROCESS
// ---------------------------------------------------------------------------
else {
  // Load the embedding model once per worker
  let embedder = null;
  let embedderPromise = null;

  async function getEmbedder() {
    if (embedder) return embedder;

    if (!embedderPromise) {
      log.info(`[Worker ${process.pid}] Loading model from ${MODELS_DIR}...`);

      embedderPromise = pipeline(
        "feature-extraction",
        path.join(".", MODEL_NAME)
      ).then(model => {
        embedder = model;
        log.info(`[Worker ${process.pid}] Model loaded`);
        return model;
      });
    }

    return embedderPromise;
  }

  const app = express();
  app.use(express.json({ limit: "10mb" }));

  app.post("/api/embedding", async (req, res) => {
    try {
      const text = req.body?.text;
      if (!text) {
        return res.status(400).json({ error: "Missing 'text'" });
      }

      const start = Date.now();
      const model = await getEmbedder();
      const result = await model(text, { pooling: "mean", normalize: true });
      const embedding = Array.from(result.data);

      res.json({ embedding, worker: process.pid, took: Date.now() - start });
    } catch (err) {
      console.error(`[Worker ${process.pid}] Error:`, err);
      res.status(500).json({ error: err.message });
    }
  });

  app.listen(PORT, () => {
    log.info(`[Worker ${process.pid}] Listening on port ${PORT}`);
  });
}

import cluster from "cluster";
import os from "os";
import express from "express";
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
  // Lazy-loaded sentence-transformers model
  let model = null;
  let loadPromise = null;

  async function getModel() {
    if (model) return model;

    if (!loadPromise) {
      // Dynamic import avoids loading on master
      const { SentenceTransformer } = await import("sentence-transformers");

      log.info(
        `[Worker ${process.pid}] Loading model '${MODEL_NAME}' from ${MODELS_DIR}`
      );

      loadPromise = SentenceTransformer.load(MODEL_NAME, {
        cacheDir: MODELS_DIR, // Load local model
      }).then((m) => {
        model = m;
        log.info(`[Worker ${process.pid}] Model loaded`);
        return model;
      });
    }

    return loadPromise;
  }

  const app = express();
  app.use(express.json({ limit: "10mb" }));

  // -------------------------------------------------------------------------
  // POST /api/embedding
  // -------------------------------------------------------------------------
  app.post("/api/embedding", async (req, res) => {
    try {
      const text = req.body?.text;
      if (!text) {
        return res.status(400).json({ error: "Missing 'text'" });
      }

      const m = await getModel();

      const start = Date.now();
      const embedding = await m.embed(text); // returns Float32Array
      const took = Date.now() - start;

      res.json({
        embedding: Array.from(embedding),
        worker: process.pid,
        took,
      });
    } catch (err) {
      console.error(`[Worker ${process.pid}] Error:`, err);
      res.status(500).json({ error: err.message });
    }
  });

  // -------------------------------------------------------------------------
  // START SERVER
  // -------------------------------------------------------------------------
  app.listen(PORT, () => {
    log.info(`[Worker ${process.pid}] Listening on port ${PORT}`);
  });
}

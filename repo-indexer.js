import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { embedText, getOsClient, fileChecksum, runLimited, INDEX_NAME, PATH_TO_REPO } from "./utils.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// LOGGER SETUP
// ---------------------------------------------------------------------------
import createLogger from "./logger.js";
const log = createLogger(import.meta.url);
// ---------------------------------------------------------------------------

async function getFiles(dir, exts = [".js", ".ts", ".java", ".py"]) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(entry => {
    const res = path.resolve(dir, entry.name);
    return entry.isDirectory() ? getFiles(res, exts) : res;
  }));
  return Array.prototype.concat(...files).filter(f => exts.includes(path.extname(f)));
}

async function getStoredChecksum(filePath) {
  try {
    const res = await getOsClient().get({
      index: INDEX_NAME,
      id: filePath
    });
    return res._source.checksum;
  } catch (err) {
    // Not found
    return null;
  }
}

async function indexRepo(baseDir) {
  let startTS = Date.now();
  log.info(`Starting indexing of repo at ${baseDir} ...`);
  const files = await getFiles(baseDir);
  log.info(`Found ${files.length} files to process. It took ${(Date.now() - startTS)}ms to scan.`);

  // Prepare tasks
  startTS = Date.now();
  log.info("Parallel tasks preparation started...");
  const tasks = files.map(f => async () => {
    const content = await fs.promises.readFile(f, "utf8");
    const checksum = fileChecksum(content);
    const existing = await getStoredChecksum(f);

    if (existing === checksum) {
      log.info(`Skipping unchanged file: ${f}`);
      return;
    }

    const emb = await embedText(content);

    await getOsClient().index({
      index: INDEX_NAME,
      id: f,
      body: {
        filename: path.basename(f),
        filepath: f,
        language: path.extname(f),
        content,
        checksum,
        embedding: emb
      }
    });
  });
  log.info(`Prepared ${tasks.length} tasks in ${(Date.now() - startTS)}ms`);

  log.info("Starting indexing...");

  // Run with parallelism = 4
  startTS = Date.now();
  await runLimited(tasks, 4);
  const duration = Date.now() - startTS;
  log.info(`Indexing completed in ${duration > 1000 ? duration/1000 + 'sec' : ducration + ' ms'}`);
}

await indexRepo(PATH_TO_REPO);

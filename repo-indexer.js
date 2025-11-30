import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { embedText, getOsClient, fileChecksum, runLimited, percentile, INDEX_NAME, PATH_TO_REPO } from "./utils.js";

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
    const res = await getOsClient().search({
      index: INDEX_NAME,
      body: {
        query: {
          ids: {
            values: [filePath]
          }
        }
      }
    });
    const hit = res.body?.hits?.hits?.[0];
    return hit?._source?.checksum ?? null;
  } catch (err) {
    // Not found
    return null;
  }
}

async function indexRepo(baseDir) {
  let totalFiles = 0;
  let embedCalls = 0;
  let embedLatencies = [];
  let skippedDuplicates = 0;

  let startTS = Date.now();
  log.info(`Starting indexing of repo at ${baseDir} ...`);
  const files = await getFiles(baseDir);
  log.info(`Found ${files.length} files to process. It took ${(Date.now() - startTS)}ms to scan.`);

  // Prepare tasks
  startTS = Date.now();
  log.info("Parallel tasks preparation started...");
  const tasks = files.map(f => async () => {
    totalFiles++;
  
    // Convert absolute path -> relative repo path with forward slashes
    const relPath = path.relative(baseDir, f).replace(/\\/g, "/");
  
    const content = await fs.promises.readFile(f, "utf8");
    const checksum = fileChecksum(content);
    const existing = await getStoredChecksum(relPath);
  
    if (existing === checksum) {
      skippedDuplicates++;
      return;
    }
  
    const start = Date.now();
    const emb = await embedText(content);
  
    embedCalls++;
    embedLatencies.push(Date.now() - start);
  
    await getOsClient().index({
      index: INDEX_NAME,
      id: relPath,
      body: {
        filename: path.basename(f),
        filepath: relPath,
        language: path.extname(f),
        content,
        checksum,
        embedding: emb
      }
    });
  });
  log.info(`Prepared ${tasks.length} tasks in ${(Date.now() - startTS)}ms`);

  log.info("Starting indexing...");

  // Run with parallelism = 2
  let startTSmain = Date.now();
  await runLimited(tasks, 2);
  const duration = Date.now() - startTSmain;
  log.info(`Indexing completed in ${duration > 1000 ? duration/1000 + ' sec' : duration + ' ms'}`);

  embedLatencies.sort((a, b) => a - b);
  const min = embedLatencies[0] ?? 0;
  const max = embedLatencies[embedLatencies.length - 1] ?? 0;
  const p50 = percentile(embedLatencies, 50);

  log.info("\n===== INDEXING STATS =====" +
  `\nFiles processed:          ${totalFiles}` +
  `\nEmbedding calls:          ${embedCalls}` +
  `\nDuplicates skipped:       ${skippedDuplicates}` +
  `\nMin latency:              ${min} ms` +
  `\nMax latency:              ${max} ms` +
  `\nP50 latency:              ${p50} ms` +
  `\nTotal indexing time:      ${duration > 1000 ? (duration/1000 + ' sec') : (duration + ' ms')}` +
  `\n==========================`);
}

await indexRepo(PATH_TO_REPO);

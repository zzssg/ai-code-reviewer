import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { embedText, getOsClient, fileChecksum, INDEX_NAME, PATH_TO_REPO } from "./utils.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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
  console.log(`Starting indexing of repo at ${baseDir} ...`);
  const files = await getFiles(baseDir);
  console.log(`Found ${files.length} files to process. It took ${(Date.now() - startTS)}ms to scan.`);

  // Prepare tasks
  startTS = Date.now();
  console.log("Parallel tasks preparation started...");
  const tasks = files.map(f => async () => {
    const content = await fs.promises.readFile(f, "utf8");
    const checksum = fileChecksum(content);
    const existing = await getStoredChecksum(f);

    if (existing === checksum) {
      console.log(`Skipping unchanged file: ${f}`);
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
  console.log(`Prepared ${tasks.length} tasks in ${(Date.now() - startTS)}ms`);

  console.log("Starting indexing...");

  // Run with parallelism = 4
  startTS = Date.now();
  await runLimited(tasks, 4);
  console.log(`Indexing completed in ${(Date.now() - startTS)}ms`);
}

await indexRepo(PATH_TO_REPO);

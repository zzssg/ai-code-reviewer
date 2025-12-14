import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  embedText,
  getOsClient,
  fileChecksum,
  runLimited,
  percentile,
  INDEX_NAME,
  PATH_TO_REPO
} from "./utils.js";

import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

import { parse as parseJava } from "java-parser";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// -----------------------------
// BLACKLIST FILENAMES
// -----------------------------
const blackList = [
  "bundle.js",
  "README.md",
  "package-info.java",
  "jquery-slim.min.js",
  "bootstrap.min.js",
  "LICENSE",

];

// -----------------------------
// FILE COLLECTION
// -----------------------------
async function getFiles(dir, exts = [".js", ".ts", ".java", ".py"]) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(entry => {
    const res = path.resolve(dir, entry.name);
    return entry.isDirectory() ? getFiles(res, exts) : res;
  }));
  return Array.prototype
    .concat(...files)
    .filter(f => exts.includes(path.extname(f)))
    .filter(f => !blackList.includes(path.basename(f)));
}

// -----------------------------
// JAVA METHOD CHUNKING
// -----------------------------
function chunkJavaByMethods(content) {
  try {
    const cst = parseJava(content);
    const lines = content.split(/\r?\n/);
    const chunks = [];

    const methodRegex = /(?:public|protected|private|static|\s)+[\w<>\[\]]+\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{/;
    let chunkIndex = 0;
    let current = null;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(methodRegex);
      if (m) {
        if (current) {
          current.end = i - 1;
          const body = lines.slice(current.start, current.end + 1).join("\n");
          chunks.push({
            chunk_id: chunkIndex++,
            function_name: current.name || "anonymous",
            start_line: current.start + 1,
            end_line: current.end + 1,
            text: body
          });
        }
        current = { start: i, name: m[1] || "anonymous" };
      }
    }

    if (current) {
      current.end = lines.length - 1;
      const body = lines.slice(current.start, current.end + 1).join("\n");
      chunks.push({
        chunk_id: chunkIndex++,
        function_name: current.name || "anonymous",
        start_line: current.start + 1,
        end_line: current.end + 1,
        text: body
      });
    }

    if (chunks.length === 0) {
      chunks.push({
        chunk_id: 0,
        function_name: "full_file",
        start_line: 1,
        end_line: lines.length,
        text: content
      });
    }

    return chunks;
  } catch (err) {
    const lines = content.split(/\r?\n/);
    return [{
      chunk_id: 0,
      function_name: "full_file",
      start_line: 1,
      end_line: lines.length,
      text: content
    }];
  }
}

// -----------------------------
// OTHER LANGUAGES CHUNKING
// -----------------------------
function chunkByFunctions(content, language) {
  const lines = content.split(/\r?\n/);
  const chunks = [];
  let current = null;
  const regexes = [];

  if (language === ".js" || language === ".ts") {
    regexes.push(/^function\s+([a-zA-Z0-9_]+)\s*\(/);
    regexes.push(/^[a-zA-Z0-9_]+\s*=\s*\([^)]*\)\s*=>/);
    regexes.push(/^class\s+([A-Za-z0-9_]+)/);
    regexes.push(/^[A-Za-z0-9_]+\([^)]*\)\s*\{/);
  } else if (language === ".py") {
    regexes.push(/^def\s+([A-Za-z0-9_]+)\s*\(/);
    regexes.push(/^class\s+([A-Za-z0-9_]+)/);
  } else {
    regexes.push(/.*/);
  }

  let chunkIndex = 0;
  const closeChunk = (endIdx) => {
    if (!current) return;
    const { start, functionName } = current;
    const body = lines.slice(start, endIdx + 1).join("\n");
    chunks.push({
      chunk_id: chunkIndex++,
      function_name: functionName,
      start_line: start + 1,
      end_line: endIdx + 1,
      text: body
    });
    current = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let matched = false;
    for (const r of regexes) {
      const m = line.match(r);
      if (m) {
        matched = true;
        closeChunk(i - 1);
        current = { start: i, functionName: m[1] || "anonymous" };
        break;
      }
    }
    if (matched) continue;
    if (i === lines.length - 1) closeChunk(i);
  }

  if (chunks.length === 0) {
    chunks.push({
      chunk_id: 0,
      function_name: "full_file",
      start_line: 1,
      end_line: lines.length,
      text: content
    });
  }

  return chunks;
}

// -----------------------------
// FILE IMPORTANCE
// -----------------------------
function computeImportance(filepath, content) {
  let score = 1.0;
  const p = filepath.toLowerCase();
  if (p.includes("/src/") || p.includes("/src/main/") || p.includes("/lib/")) score += 1.0;
  if (p.includes("/test/") || p.includes("/spec/") || p.includes("/__tests__/")) score -= 0.5;
  const lines = content.split(/\r?\n/).length;
  if (lines < 200) score += 0.5;
  return Math.max(score, 0.1);
}

// -----------------------------
// GET STORED CHECKSUMS
// -----------------------------
async function getStoredFileChecksum(docId) {
  try {
    const res = await getOsClient().get({ index: INDEX_NAME, id: docId });
    return res.body?._source?.checksum ?? null;
  } catch {
    return null;
  }
}

// -----------------------------
// MAIN INDEXING
// -----------------------------
async function indexRepo(baseDir) {
  let totalChunks = 0;
  let embedCalls = 0;
  let embedLatencies = [];
  let skippedDuplicates = 0;

  log.info(`Starting indexing of repo at ${baseDir} ...`);
  const files = await getFiles(baseDir);
  log.info(`Found ${files.length} files to process.`);

  const tasks = files.map(f => async () => {
    const relPath = path.relative(baseDir, f).replace(/\\/g, "/");
    const language = path.extname(f);
    const content = await fs.promises.readFile(f, "utf8");
    const fileHash = fileChecksum(content);

    // Skip unchanged files entirely
    const existingFileChecksum = await getStoredFileChecksum(relPath);
    if (existingFileChecksum === fileHash) {
      skippedDuplicates++;
      return; // do not process chunks or embeddings
    }

    // Chunk detection
    let chunks = [];
    if (language === ".java") chunks = chunkJavaByMethods(content);
    else chunks = chunkByFunctions(content, language);

    const importance = computeImportance(relPath, content);

    // Store parent file doc
    await getOsClient().index({
      index: INDEX_NAME,
      id: relPath,
      body: {
        doc_type: "file",
        filename: path.basename(f),
        filepath: relPath,
        language,
        checksum: fileHash,
        importance,
        chunks_count: chunks.length
      }
    });

    // Process chunks (we know file is new/changed, so all chunks need embedding)
    for (const chunk of chunks) {
      totalChunks++;
      const chunkId = `${relPath}::chunk_${chunk.chunk_id}`;
      const chunkHash = fileChecksum(chunk.text);

      const t0 = Date.now();
      const embeddings = await embedText(chunk.text);
      embedCalls++;
      embedLatencies.push(Date.now() - t0);

      if (embeddings.length <= 1) {
        await getOsClient().index({
          index: INDEX_NAME,
          id: chunkId,
          body: {
            doc_type: "chunk",
            filename: path.basename(f),
            filepath: relPath,
            language,
            chunk_id: chunk.chunk_id,
            function_name: chunk.function_name,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            content: chunk.text,
            checksum: chunkHash,
            embedding: embeddings[0],
            importance
          }
        });
      } else {
        for (let i = 0; i < embeddings.length; i++) {
          const subId = `${chunkId}::sub_${i}`;
          await getOsClient().index({
            index: INDEX_NAME,
            id: subId,
            body: {
              doc_type: "chunk",
              filename: path.basename(f),
              filepath: relPath,
              language,
              chunk_id: chunk.chunk_id,
              sub_idx: i,
              function_name: chunk.function_name,
              start_line: chunk.start_line,
              end_line: chunk.end_line,
              content: chunk.text,
              checksum: chunkHash,
              embedding: embeddings[i],
              importance
            }
          });
        }
      }
    }
  });

  log.info(`Prepared ${tasks.length} tasks. Starting indexing...`);
  const started = Date.now();
  await runLimited(tasks, 2);
  const duration = Date.now() - started;

  embedLatencies.sort((a, b) => a - b);
  const min = embedLatencies[0] ?? 0;
  const max = embedLatencies[embedLatencies.length - 1] ?? 0;
  const p50 = percentile(embedLatencies, 50);

  log.info(
    "\n===== INDEXING STATS =====" +
    `\nChunks processed:         ${totalChunks}` +
    `\nEmbedding calls:          ${embedCalls}` +
    `\nDuplicates skipped:       ${skippedDuplicates}` +
    `\nMin latency:              ${min} ms` +
    `\nMax latency:              ${max} ms` +
    `\nP50 latency:              ${p50} ms` +
    `\nTotal indexing time:      ${duration > 1000 ? (duration/1000 + ' sec') : (duration + ' ms')}` +
    `\n==========================`
  );
}

// run
await indexRepo(PATH_TO_REPO);

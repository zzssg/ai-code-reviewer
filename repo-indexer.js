import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Client } from "@opensearch-project/opensearch";
import { embedText, getOsClient, INDEX_NAME, PATH_TO_REPO } from "./utils.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function getFiles(dir, exts = [".js", ".ts", ".java", ".py"]) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(entry => {
    const res = path.resolve(dir, entry.name);
    return entry.isDirectory() ? getFiles(res, exts) : res;
  }));
  return Array.prototype.concat(...files).filter(f => exts.includes(path.extname(f)));
}

async function indexRepo(baseDir) {
  const files = await getFiles(baseDir);
  for (const f of files) {
    const content = await fs.promises.readFile(f, "utf8");
    const emb = await embedText(content);
    console.log(`Indexing file ${path.basename(f)} of size ${content.length}`);
    await getOsClient().index({
      index: INDEX_NAME,
      body: {
        filename: path.basename(f),
        filepath: f,
        language: path.extname(f),
        content,
        embedding: emb
      }
    });
  }
  console.log(`Indexed repo: ${files.length} files`);
}

await indexRepo(PATH_TO_REPO);

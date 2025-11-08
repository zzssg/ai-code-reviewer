import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function getFiles(dir, exts = [".js", ".ts", ".java", ".py"]) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(entry => {
    const res = path.resolve(dir, entry.name);
    return entry.isDirectory() ? getFiles(res, exts) : res;
  }));
  return Array.prototype.concat(...files).filter(f => exts.includes(path.extname(f)));
}

async function embedText(text) {
  const response = await fetch("http://localhost:11434/api/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding request failed: ${response.status} ${errText}`);
  }

  const data = await response.json();

  // Assuming API returns: { embedding: [ ...numbers... ] }
  if (!data.embedding) {
    throw new Error("Invalid response: missing 'embedding' field");
  }

  return data.embedding;
}

async function indexRepo(baseDir) {
  const files = await getFiles(baseDir);
  for (const f of files) {
    const content = await fs.promises.readFile(f, "utf8");
    const emb = await embedText(content);
    await osClient.index({
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

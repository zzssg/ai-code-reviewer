// pr-reviewer.js
import express from "express";
import axios from "axios";
import { getOsClient, queryLLM } from "./utils.js";
import { embedText } from "./utils.js";
import parseDiff from "parse-diff";

import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

const app = express();
app.use(express.json());

/**
 * Parse unified diff text (git / bitbucket) and return per-file hunks:
 * [{ filepath, hunks: [{ newStart, newLines, newEnd }, ...] }, ...]
 */
function extractHunksFromDiff(diffText) {
  const files = parseDiff(diffText); // parse-diff returns array of files with hunks
  const out = [];

  for (const f of files) {
    // choose destination path (new file name). parse-diff sets to f.to
    const filepath = f.to || f.from;
    if (!filepath) continue;
    const hunks = (f.chunks || []).map(h => {
      // each chunk has newStart and newLines
      const newStart = h.newStart;
      const newLines = h.newLines;
      const newEnd = newStart + (newLines || 0) - 1;
      return { newStart, newEnd, lines: newLines };
    });
    out.push({ filepath, hunks });
  }
  return out;
}

/**
 * Get candidate chunk docs for a given filepath and hunk range.
 * We query documents with id prefix filepath::chunk_ or filepath::chunk_X::sub_Y
 * and filter those whose start_line <= hunk_end && end_line >= hunk_start.
 * Returns array of docs { _id, _source }.
 */
async function getCandidateChunksForHunk(filepath, hunkStart, hunkEnd, os) {
  // Query by filepath prefix and line overlap.
  // OpenSearch: search for docs with filepath == filepath and line overlap.
  const q = {
    bool: {
      must: [
        { term: { filepath } },
        {
          bool: {
            should: [
              { range: { start_line: { lte: hunkEnd } } },
              { range: { end_line: { gte: hunkStart } } }
            ]
          }
        }
      ]
    }
  };

  const res = await os.search({
    index: process.env.INDEX_NAME || "repo-code-embeddings",
    body: {
      size: 200, // fetch a reasonable candidate set
      query: q,
      _source: ["filepath", "filename", "start_line", "end_line", "function_name", "content", "embedding", "importance"]
    }
  });

  return (res.body.hits.hits || []).map(h => ({ id: h._id, source: h._source, score: h._score }));
}

/**
 * Cosine similarity utility for vectors (assumes float arrays)
 */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
function cosine(a, b) {
  const n1 = norm(a);
  const n2 = norm(b);
  if (n1 === 0 || n2 === 0) return 0;
  return dot(a, b) / (n1 * n2);
}

/**
 * Search context: produce relevant chunk-level contexts for given query text (description+diff).
 * We:
 *  - obtain query embeddings (array)
 *  - for every changed file/hunk, fetch candidate chunks that overlap the hunk,
 *    compute cosine similarity locally for each candidate using each query vector,
 *    multiply by importance, and aggregate top results.
 *  - if no hunks matched, we fallback to global knn search using the first query vector.
 */
async function searchContext(queryText, parsedDiff) {
  const os = getOsClient();
  const queryEmbeddings = await embedText(queryText); // array of vectors

  let allMatches = [];

  // If parsedDiff provided, prioritize hunks
  if (parsedDiff && parsedDiff.length > 0) {
    for (const fileDiff of parsedDiff) {
      const filepath = fileDiff.filepath.replace(/^\//, "");
      for (const hunk of fileDiff.hunks) {
        const hStart = hunk.newStart;
        const hEnd = hunk.newEnd;

        // fetch candidate docs overlapping this hunk
        const candidates = await getCandidateChunksForHunk(filepath, hStart, hEnd, os);

        // compute similarity for each candidate against each query vector
        for (const cand of candidates) {
          const emb = cand.source.embedding;
          if (!Array.isArray(emb) || emb.length === 0) continue;
          for (const qv of queryEmbeddings) {
            const sim = cosine(emb, qv);
            const weighted = sim * (cand.source.importance ?? 1.0);
            allMatches.push({
              score: weighted,
              filepath: cand.source.filepath,
              function_name: cand.source.function_name,
              start_line: cand.source.start_line,
              end_line: cand.source.end_line,
              content: cand.source.content,
              importance: cand.source.importance ?? 1.0
            });
          }
        }
      }
    }
  }

  // If no matches were found (or too few), fallback to global knn on embeddings
  if (allMatches.length < 5) {
    // Use the first query embedding for fallback knn
    const qv = queryEmbeddings[0];
    const res = await os.search({
      index: process.env.INDEX_NAME || "repo-code-embeddings",
      body: {
        size: 50,
        query: {
          knn: {
            embedding: {
              vector: qv,
              k: 50
            }
          }
        },
        _source: ["filepath", "filename", "start_line", "end_line", "function_name", "content", "embedding", "importance"]
      }
    });

    for (const hit of res.body.hits.hits) {
      const s = hit._source;
      const sim = cosine(qv, s.embedding || []);
      const weighted = sim * (s.importance ?? 1.0);
      allMatches.push({
        score: weighted,
        filepath: s.filepath,
        function_name: s.function_name,
        start_line: s.start_line,
        end_line: s.end_line,
        content: s.content,
        importance: s.importance ?? 1.0
      });
    }
  }

  // Sort, dedupe and return top N
  allMatches.sort((a, b) => b.score - a.score);

  const seen = new Set();
  const unique = [];
  for (const m of allMatches) {
    const key = `${m.filepath}:${m.start_line}-${m.end_line}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(m);
    }
    if (unique.length >= 10) break;
  }

  return unique;
}

/**
 * Format context snippets (with line numbers) for LLM
 */
function buildContextText(matches) {
  return matches.map(m => {
    const snippet = m.content.split("\n").slice(m.start_line - 1, m.end_line).join("\n");
    return `File: ${m.filepath}
Lines: ${m.start_line}-${m.end_line}
Function: ${m.function_name}
Importance: ${m.importance}
---
${snippet}
`;
  }).join("\n\n");
}

async function reviewPullRequest({ description, diff, parsedDiff, contextMatches }) {
  const contextText = buildContextText(contextMatches || []);
  const prompt = `
You are a senior code reviewer.

Repository context (relevant snippets):
${contextText}

Pull request description:
${description}

Code diff:
${diff}

Please provide a detailed review with:
- potential bugs or logic errors
- code quality or readability issues
- suggestions for improvement
`;

  return queryLLM(prompt);
}

// endpoint
app.post("/bitbucket/pr-event", async (req, res) => {
  try {
    const event = req.body;
    const pr = event.pullrequest;
    const prId = pr.id;
    const description = pr.description || "";
    const diffUrl = pr.links.diff.href;

    // fetch diff
    const diffRes = await axios.get(diffUrl);
    const diff = diffRes.data;

    // parse diff into hunks
    const parsed = parseDiff(diff).map(f => ({
      filepath: f.to || f.from,
      hunks: (f.chunks || []).map(h => ({
        newStart: h.newStart,
        newEnd: h.newStart + (h.newLines || 0) - 1
      }))
    }));

    // create combined query: description + diff
    const queryText = description + "\n" + diff;

    // search with semantic diff chunking prioritized
    const contextMatches = await searchContext(queryText, parsed);

    // ask LLM
    const review = await reviewPullRequest({ description, diff, parsedDiff: parsed, contextMatches });

    // post back to Bitbucket
    await axios.post(
      `https://api.bitbucket.org/2.0/repositories/${pr.destination.repository.full_name}/pullrequests/${prId}/comments`,
      { content: { raw: review } },
      { headers: { Authorization: `Bearer ${process.env.BITBUCKET_TOKEN}` } }
    );

    res.sendStatus(200);
  } catch (err) {
    log.error("PR handler error", err);
    res.status(500).send({ error: String(err) });
  }
});

app.listen(3001, () => log.info("PR Review Bot listening on :3001"));

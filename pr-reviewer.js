import express from "express";
import axios from "axios";
import path from "path";
import { getOsClient, embedText, queryLLM, INDEX_NAME } from "./utils.js";

import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

const PORT = process.env.PORT || 3001;
const BITBUCKET_TOKEN = process.env.BITBUCKET_TOKEN;
// IMPORTANT: make this the API root, not including "/projects/..."
const BITBUCKET_BASE = process.env.BITBUCKET_BASE || "http://bitbucket:7990/rest/api/1.0";

if (!BITBUCKET_TOKEN) {
  log.error("BITBUCKET_TOKEN environment variable is not set");
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: "10mb" }));

const TOP_K = 10;

/**
 * Convert Bitbucket diff → hunks containing ONLY changed lines.
 *
 * Rule:
 *   If REMOVED and ADDED exist for the same destination line → keep ADDED only.
 */
function transformBitbucketDiff(diffResponse) {
  if (!diffResponse || !Array.isArray(diffResponse.diffs)) {
    return [];
  }

  const result = [];

  for (const file of diffResponse.diffs) {
    const filepath =
      file?.destination?.toString || file?.source?.toString || "unknown";
    const hunksOut = [];

    if (!Array.isArray(file.hunks)) {
      result.push({ filepath, hunks: [] });
      continue;
    }

    for (const hunk of file.hunks) {
      //
      // STEP 1 — Collect ADDED and REMOVED lines grouped by destination
      //
      const addedByDest = new Map();
      const removedByDest = new Map();

      for (const seg of hunk.segments || []) {
        const isAdded = seg.type === "ADDED";
        const isRemoved = seg.type === "REMOVED";

        if (!isAdded && !isRemoved) continue;

        for (const line of seg.lines || []) {
          const dest = line.destination;
          const src = line.source;

          if (isAdded && typeof dest === "number") {
            addedByDest.set(dest, dest);
          }

          // REMOVED: grouped by destination if valid, else fallback to source
          if (isRemoved) {
            if (typeof dest === "number") {
              removedByDest.set(dest, dest);
            } else if (typeof src === "number") {
              removedByDest.set(src, src);
            }
          }
        }
      }

      //
      // STEP 2 — Apply rule: If ADDED exists for same destination → remove REMOVED
      //
      for (const dest of addedByDest.keys()) {
        if (removedByDest.has(dest)) {
          removedByDest.delete(dest); // ADDED wins
        }
      }

      //
      // STEP 3 — Collect final changed line numbers
      //
      const changedLines = [
        ...Array.from(addedByDest.values()),
        ...Array.from(removedByDest.values())
      ];

      if (changedLines.length === 0) continue;

      changedLines.sort((a, b) => a - b);

      //
      // STEP 4 — Merge adjacent changed lines into hunks
      //
      let start = changedLines[0];
      let prev = start;

      for (let i = 1; i < changedLines.length; i++) {
        const cur = changedLines[i];
        if (cur !== prev + 1) {
          hunksOut.push({
            start,
            end: prev,
            code: extractChangedCode(hunk, start, prev)
       });
          start = cur;
        }
        prev = cur;
      }

      hunksOut.push({
        start,
        end: prev,
        code: extractChangedCode(hunk, start, prev)
      });
    }

    result.push({ filepath, hunks: hunksOut });
  }

  return result;
}

function extractChangedCode(hunk, start, end) {
  const lines = [];

  for (const seg of hunk.segments || []) {
    const isAdded = seg.type === "ADDED";
    const isRemoved = seg.type === "REMOVED";
    if (!isAdded && !isRemoved) continue;

    for (const ln of seg.lines || []) {
      const dest = ln.destination ?? ln.source;
      if (typeof dest !== "number") continue;
      if (dest < start || dest > end) continue;

      lines.push({
        type: isAdded ? "ADDED" : "REMOVED",
        line: ln.line
      });
    }
  }

  return lines;
}



/**
 * Build a natural-language query for each changed hunk.
 *
 * @param {Array} transformed   Output from transformBitbucketDiff()
 * @param {Object} diffResponse Raw Bitbucket diff response (for code extraction)
 * @param {Number} contextLines How many context lines to include around change
 */
function buildHunkQueries(transformed, diffResponse, contextLines = 5) {
  const queries = [];

  const getFileDiff = filepath =>
    diffResponse.diffs.find(d =>
      (d.destination?.toString || d.source?.toString) === filepath
    );

  for (const file of transformed) {
    const filepath = file.filepath;
    const fileDiff = getFileDiff(filepath);
    if (!fileDiff) continue;

    for (const h of file.hunks) {
      const snippet = extractSnippet(fileDiff, h.start, h.end, contextLines);
      const changes = h.code.map(l => `${l.type === "ADDED" ? "+" : "-"}${l.line}`).join("\n");

      const query = [
        `File: ${filepath.split("/").pop()}`,
        `Changed code around line ${h.start}:`,
        snippet || "<no code extracted>",
        ``,
        `Find: methods using these fields, methods calling this code, code with similar patterns.`
      ].join("\n");

      queries.push({ filepath, start: h.start, end: h.end, changes: changes, query });
    }
  }

  return queries;
}

/**
 * Extracts code around given changed lines from Bitbucket diff hunk structure.
 *
 * Includes ADDED / REMOVED / CONTEXT but restricts the window.
 */
function extractSnippet(fileDiff, startLine, endLine, contextLines) {
  let lines = [];

  for (const hunk of fileDiff.hunks || []) {
    for (const seg of hunk.segments || []) {
      for (const line of seg.lines || []) {
        // Use destination if available, else source
        const lineNum =
          typeof line.destination === "number"
            ? line.destination
            : typeof line.source === "number"
            ? line.source
            : null;

        if (lineNum == null) continue;

        const minRange = startLine - contextLines;
        const maxRange = endLine + contextLines;

        if (lineNum >= minRange && lineNum <= maxRange) {
          const prefix =
            seg.type === "ADDED"
              ? "+"
              : seg.type === "REMOVED"
              ? "-"
              : " ";

          lines.push(prefix + line.line);
        }
      }
    }
  }

  return lines.join("\n");
}

export async function makeEmbeddingsFromQueries(queries) {
  return Promise.all(
    queries.map(q => embedText(q.query).then(embedding => ({
      ...q,
      embedding: embedding[0] || []
    })))
  );
}

/**
 * Search OpenSearch by vector.
 *
 * Expects documents structured with a field "embedding" containing numeric arrays.
 */
export async function searchOpenSearch(embeddingVector, topK = TOP_K) {
  try {
    const knnResp = await getOsClient().search({
      index: INDEX_NAME,
      body: {
        size: topK,
        query: {
          knn: {
            embedding: {
              vector: embeddingVector,
              k: topK
            }
          }
        }
      }
    });

    if (knnResp.body?.hits?.hits?.length) {
      // Deduplicate by 'checksum' field
      const rawValues = knnResp.body.hits.hits.map(h => ({ _id: h._id, score: h._score, source: h._source }));
      return Array.from(new Map(rawValues.map(h => [h.source.checksum, h])).values()).slice(0, 3);
    }
  } catch (err) {
    console.log(`Error while querying Opensearch with knn: ${err.message}`);
  }
}

function buildHunkSummaryPrompt(bundle) {
  const sb = [];
  sb.push(`Filepath: ${bundle.filepath}`);
  sb.push(`Changed lines: ${bundle.start}-${bundle.end}`);
  sb.push(`Changed code: \n${bundle.changes}\n`);
  sb.push(`Contextual retrievals (top ${bundle.results.length}):`);
  for (const r of bundle.results) {
    sb.push(`--- ${r.source.filepath} ${r.source.start_line || ""}-${r.source.end_line || ""}\n${r.source.content}`);
  }
  sb.push(`\nTask: Summarize the purpose of the change and list potential problems, edge cases, or missing considerations. Keep it concise (3-6 bullet points).\n\n`);
  return sb.join("\n");
}

async function buildFinalReviewPrompt(prMeta, diff) {
  const transformedDiff = transformBitbucketDiff(diff);
  const queries = buildHunkQueries(transformedDiff, diff);

  const embeddings = await makeEmbeddingsFromQueries(queries);
  const hunkSummaries = [];
  for (const q of embeddings) {
    const results = (await searchOpenSearch(q.embedding, 5)).map(r => {
      const {embedding, ...filteredSource} = r.source; 
      return { id: r._id, score: r.score, source: filteredSource };
    });
    const { embedding, ...filteredQ } = q;
    hunkSummaries.push({
      filename: filteredQ.filepath,
      hunk: { start: filteredQ.start, end: filteredQ.end }, 
      summary: buildHunkSummaryPrompt({ ...filteredQ, results })
    });
  }
  const sb = [];
  sb.push(`You are a senior reviewer. Produce a concise PR overview.`);
  sb.push(`Given the PR metadata, hunks, and context, output the following (strict limits):\n`);
  sb.push(`Purpose (1–2 short sentences)\n`);
  sb.push(`Key issues (3–5 bullets, each max 1 line)\n`);
  sb.push(`Verdict: “✅Approve” or “❌Request changes” + 1 line reason\n`);
  sb.push(`Do NOT include long explanations, code blocks, or restatements of code.`);
  sb.push(`Keep the entire answer under ~700 characters.\n`);
  sb.push(`You are an expert pull-request reviewer. Review the following pull request composed of several hunks.\n`);
  sb.push(`PR metadata: ${JSON.stringify(prMeta)}\n`);
  sb.push(`Hunk summaries and contextual reasons:\n`);
  for (const hs of hunkSummaries) {
    sb.push(`Hunk #${hunkSummaries.indexOf(hs) + 1}:`);
    sb.push(`Filename: ${path.basename(hs.filename)}`);
    sb.push(`${hs.summary}`);
  }
  return sb.join("\n");
}

/* ---------------------------
   HTTP handler for Bitbucket PR events
   --------------------------- */

app.post("/bitbucket/pr-event", async (req, res) => {
  const PR_EVENTS_SUPPORTED = new Set(["pr:opened", "pr:updated", "pr:reopened"]);
  try {
    log.info(`Received PR event: ${JSON.stringify(req.body)}`);

    const event = req.body;
    const eventKey = event.eventKey;
    if (!PR_EVENTS_SUPPORTED.has(eventKey)) {
      log.info(`Ignoring unsupported event key "${eventKey}"`);
      return res.sendStatus(200);
    }

    const pr = event.pullRequest;
    if (!pr) {
      log.warn("No pullRequest object in webhook payload");
      return res.sendStatus(400);
    }

    const prId = pr.id;
    // prefer fromRef repository (event uses fromRef / toRef)
    const sourceRepo = pr.fromRef?.repository;
    const destRepo = pr.toRef?.repository || pr.destination?.repository;
    const repoSlug = sourceRepo?.slug || destRepo?.slug;
    const projectKey = sourceRepo?.project?.key || destRepo?.project?.key;

    if (!repoSlug || !projectKey) {
      log.error("Cannot determine repoSlug or projectKey from webhook payload", { repoSlug, projectKey });
      return res.status(400).send({ error: "Missing repository information in webhook payload" });
    }

    // Build canonical endpoints using BITBUCKET_BASE
    const prDetailsUrl = `${BITBUCKET_BASE}/projects/${encodeURIComponent(projectKey)}/repos/${encodeURIComponent(repoSlug)}/pull-requests/${prId}`;
    const diffUrl = `${prDetailsUrl}/diff`;
    // optionally: activities and changes endpoints:
    const activitiesUrl = `${prDetailsUrl}/activities`;
    const changesUrl = `${prDetailsUrl}/changes`;

    // Headers for Bitbucket Server PAT or token
    const headers = { Authorization: `Bearer ${BITBUCKET_TOKEN}` };

    log.info(`Fetching PR details from: ${prDetailsUrl}`);
    const prDetailsRes = await axios.get(prDetailsUrl, { headers });
    const prDetails = prDetailsRes.data;

    // optionally log reviewers / participants
    log.info(`PR #${prId} details: title="${prDetails.title}", state=${prDetails.state}, reviewers=${(prDetails.reviewers || []).map(r=>r.user?.name).join(",")}`);

    // fetch raw diff
    log.info(`Fetching PR diff from: ${diffUrl}`);
    const diffRes = await axios.get(diffUrl, { headers });
    const diff = diffRes.data;
    log.info(`PR diff received: ${JSON.stringify(diff)}`);


    const finalReviewPrompt = await buildFinalReviewPrompt(
      {
        project: projectKey,
        repo: repoSlug,
        pr: prId,
        from: diff.fromHash,
        to: diff.toHash
      }, 
      diff
    );
    log.info(`Final review prompt built. Length: ${finalReviewPrompt.length} characters. Content: ${finalReviewPrompt}`);

    // Query LLM for review
    log.info(`Querying LLM for PR review...`);
    const review = await queryLLM(finalReviewPrompt);
    log.info(`LLM review received. Length: ${review.length} characters. Content: ${review}`);

    // post a comment back to PR (simple top-level comment)
    const postCommentUrl = `${prDetailsUrl}/comments`;
    log.info(`Posting comment back to PR #${prId} to: ${postCommentUrl}`);
    await axios.post(postCommentUrl, { text: review }, { headers });

    log.info(`Posted review comment for PR #${prId}`);
    res.sendStatus(200);
  } catch (err) {
    log.error("PR handler error", err);
    res.status(500).send({ error: String(err) });
  }
});

app.listen(PORT, () => log.info(`PR Review Bot listening on :${PORT}`));

/**
 * Review Engine — Core LLM analysis and OpenSearch persistence.
 * This module MUST NOT make any Bitbucket API calls.
 *
 * Responsibilities:
 *  - Load/save review state in OpenSearch
 *  - Parse and transform PR diffs
 *  - Build review prompts with context from code embeddings
 *  - Run LLM-based architectural, bucket, and per-file reviews
 *  - Track issue lifecycle (NEW / OPEN / RESOLVED)
 *  - Produce structured review state as the contract with comment management
 */

import * as dotenv from 'dotenv';
dotenv.config();

import {
  getAggregateSummaryPrompt,
  getArchitectureReviewPreface,
  getArchitectureReviewPrompt,
  getBucketReviewPreface,
  getFileReviewPrompt,
  getIssuesNormalizationPrompt
} from "./prompt.js";
import { DIFF_SEGMENT_TYPES } from "./types.js";
import { architecture_review_schema, pr_review_issues_schema, pr_summary_schema } from "./llm.schemas.js";
import { createLogger, embedText, ensureIndex, getOsClient, queryLLM } from "./utils.js";

const log = createLogger(import.meta.url);

const REVIEW_STATE_INDEX = "tftools-code-review-states";
// Default matches the actual file in the repository
const REVIEW_STATE_SCHEMA = process.env.EMB_OS_REVIEW_SCHEMA || "data/tftools-code-review-states.json";
const MAX_POSTED_ISSUES = 20;
const ISSUE_SEVERITY_ORDER = { "CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4 };
const TOP_K = 10;
const IGNORE_FILENAMES = ["package-lock.json", "package.json", "gradle.properties"];
const IGNORE_LINE_TOLERANCE = Number(process.env.PR_REVIEWER_ISSUE_LINE_TOLERANCE || 5);

// ---------------------------------------------------------------------------
// OpenSearch state persistence
// ---------------------------------------------------------------------------

async function ensureReviewStateIndex() {
  await ensureIndex(REVIEW_STATE_INDEX, REVIEW_STATE_SCHEMA);
}

export async function loadReviewState(reviewStateId) {
  await ensureReviewStateIndex();
  try {
    const response = await getOsClient().get({ index: REVIEW_STATE_INDEX, id: reviewStateId });
    return response?.body?._source || response?._source || null;
  } catch (error) {
    const statusCode = error?.meta?.statusCode || error?.statusCode;
    if (statusCode === 404) return null;
    throw error;
  }
}

export async function saveReviewState(reviewState) {
  await ensureReviewStateIndex();
  await getOsClient().index({
    index: REVIEW_STATE_INDEX,
    id: reviewState.reviewId,
    body: reviewState,
    refresh: true
  });
  return reviewState;
}

export function buildInitialReviewState(prMeta, reviewId, eventKey) {
  const now = new Date().toISOString();
  return {
    reviewId,
    projectKey: prMeta.project,
    repoSlug: prMeta.repo,
    prId: prMeta.pr,
    lastEventKey: eventKey,
    fromHash: prMeta.from,
    toHash: prMeta.to,
    summaryCommentId: null,
    summaryCommentVersion: null,
    openIssues: [],
    createdAt: now,
    updatedAt: now
  };
}

/**
 * Extracts bot-manageable state from a persisted review state.
 * Returns the map of unresolved issues (keyed by issueKey) and the
 * existing summary comment reference (if any).
 */
export function buildBotStateFromReviewState(reviewState) {
  const unresolvedIssues = new Map();
  const inlineComments = [];
  for (const issue of reviewState?.openIssues || []) {
    if (issue.status === "RESOLVED") continue;
    const issueData = {
      filename: issue.filename,
      line_number: issue.line_number,
      issue_description: issue.issue_description,
      suggestion: issue.suggestion || {}
    };
    const entry = {
      id: issue.inlineCommentId,
      issueKey: issue.issueKey || buildIssueKey(issueData),
      status: issue.status || "OPEN",
      filename: issue.filename,
      line_number: issue.line_number,
      issue: issueData
    };
    unresolvedIssues.set(entry.issueKey, entry);
    inlineComments.push(entry);
  }
  return {
    inlineComments,
    summaryComment: reviewState?.summaryCommentId
      ? { id: reviewState.summaryCommentId, version: reviewState.summaryCommentVersion }
      : null,
    unresolvedIssues
  };
}

/**
 * Merges the current review state with the results of a new review pass.
 * @param {object} reviewState - Existing review state document
 * @param {object} params
 * @param {string} params.eventKey
 * @param {object} params.prMeta
 * @param {Array}  params.issueStatuses - Array of {issue, status} objects
 * @param {Array}  params.resolvedOverflowIssues - Issues beyond MAX_POSTED_ISSUES
 * @param {object|null} params.summaryComment - Posted summary comment {id, version}
 */
export function mergeReviewStateWithResults(reviewState, {
  eventKey,
  prMeta,
  issueStatuses,
  resolvedOverflowIssues = [],
  summaryComment = null
}) {
  const now = new Date().toISOString();

  const openIssues = issueStatuses.map(({ issue, status }) => {
    const primaryLocation = getPrimaryLocation(issue);
    return {
      issueKey: buildIssueKey(issue),
      filename: primaryLocation?.filename || "",
      line_number: primaryLocation?.line_number ?? null,
      issue_description: issue.issue_description,
      severity: normalizeSeverity(issue.severity),
      suggestion: issue.suggestion || {},
      locations: Array.isArray(issue.locations) ? issue.locations : [],
      status,
      inlineCommentId: issue.inlineCommentId ?? null
    };
  }).concat(resolvedOverflowIssues.map(issue => {
    const primaryLocation = getPrimaryLocation(issue);
    return {
      issueKey: buildIssueKey(issue),
      filename: primaryLocation?.filename || "",
      line_number: primaryLocation?.line_number ?? null,
      issue_description: issue.issue_description,
      severity: normalizeSeverity(issue.severity),
      suggestion: issue.suggestion || {},
      locations: Array.isArray(issue.locations) ? issue.locations : [],
      status: "RESOLVED",
      inlineCommentId: null
    };
  }));

  return {
    ...reviewState,
    lastEventKey: eventKey,
    fromHash: prMeta.from,
    toHash: prMeta.to,
    summaryCommentId: summaryComment?.id ?? reviewState.summaryCommentId ?? null, 
    summaryCommentVersion: summaryComment?.version ?? reviewState.summaryCommentVersion ?? null,
    openIssues,
    updatedAt: now
  };
}

// ---------------------------------------------------------------------------
// Diff parsing and transformation
// ---------------------------------------------------------------------------

/**
 * Transforms the Bitbucket diff API response into a simplified internal format.
 * Returns [{filepath, hunks: [{start, end, code}]}]
 */
export function transformBitbucketDiff(diffResponse) {
  if (!diffResponse || !Array.isArray(diffResponse.diffs)) { 
    return [];
  }

  const result = [];

  for (const file of diffResponse.diffs) { 
    const filepath = file.destination?.toString || file.source?.toString || "unknown";
    if (IGNORE_FILENAMES.some(name => filepath.endsWith(name))) {
      log.info(`Skipping file ${filepath} due to ignore list`);
      continue;
    }

    const hunksOut = [];

    if (!Array.isArray(file.hunks)) {
      log.warn(`file.hunks is not array for ${filepath}: ${JSON.stringify(file.hunks)}`);
      result.push({ filepath, hunks: [] });
      continue;
    }

    for (const hunk of file.hunks) {
      const addedByDest = new Map();
      const removedByDest = new Map();

      for (const seg of hunk.segments || []) {
        const isAdded = seg.type === DIFF_SEGMENT_TYPES.ADDED;
        const isRemoved = seg.type === DIFF_SEGMENT_TYPES.REMOVED;

        if (!isAdded && !isRemoved) continue;

        for (const line of seg.lines || []) {
          const dest = line.destination;
          const src = line.source;

          if (isAdded && typeof dest === "number") {
            addedByDest.set(dest, dest);
          }
          if (isRemoved) {
            if (typeof dest === "number") {
              removedByDest.set(dest, dest);
            } else if (typeof src === "number") {
              removedByDest.set(src, src);
            }
          }
        }
      }

      for (const dest of addedByDest.keys()) {
        if (removedByDest.has(dest)) removedByDest.delete(dest);
      }

      const changedLines = [...Array.from(addedByDest.values()), ...Array.from(removedByDest.values())];
      if (changedLines.length === 0) continue;
      changedLines.sort((a, b) => a - b);

      let start = changedLines[0];
      let prev = start;
      for (let i = 1; i < changedLines.length; i++) {
        const cur = changedLines[i];
        if (cur !== prev + 1) {
          hunksOut.push({ start, end: prev, code: extractChangedCode(hunk, start, prev) });
          start = cur;
        }
        prev = cur;
      }
      hunksOut.push({ start, end: prev, code: extractChangedCode(hunk, start, prev) });
    }

    result.push({ filepath, hunks: hunksOut });
  }

  return result;
}

function extractChangedCode(hunk, start, end) {
  const lines = [];
  for (const seg of hunk.segments || []) {
    const isAdded = seg.type === DIFF_SEGMENT_TYPES.ADDED;
    const isRemoved = seg.type === DIFF_SEGMENT_TYPES.REMOVED;
    if (!isAdded && !isRemoved) continue;

    for (const ln of seg.lines || []) {
      const dest = ln.destination ?? ln.source;
      if (typeof dest !== "number") continue;
      if (dest < start || dest > end) continue;
      lines.push({ type: isAdded ? "ADDED" : "REMOVED", line: ln.line });
    }
  }
  return lines;
}

function buildHunkQueries(transformed, diffResponse, contextLines = 5) {
  const queries = [];
  const getFileDiff = filepath => (diffResponse.diffs || []).find(
    d => (d.destination?.toString || d.source?.toString) === filepath
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
        "",
        "Find methods using these fields, methods calling this code, code with similar patterns."
      ].join("\n");
      queries.push({ filepath, start: h.start, end: h.end, changes, query, snippet });
    }
  }

  return queries;
}

function extractSnippet(fileDiff, startLine, endLine, contextLines) {
  const lines = [];
  for (const hunk of fileDiff.hunks || []) {
    for (const seg of hunk.segments || []) {
      const isAdded = seg.type === DIFF_SEGMENT_TYPES.ADDED;
      const isRemoved = seg.type === DIFF_SEGMENT_TYPES.REMOVED;

      for (const line of seg.lines || []) {
        // Use destination-preferred number for range filtering
        const lineNum = typeof line.destination === "number"
          ? line.destination
          : typeof line.source === "number"
            ? line.source
            : null;
        if (lineNum == null) continue;

        const minRange = startLine - contextLines;
        const maxRange = endLine + contextLines;
        if (lineNum >= minRange && lineNum <= maxRange) {
          const prefix = isAdded ? "+" : isRemoved ? "-" : " ";
          // Canonical line number: destination for ADDED/CONTEXT, source for REMOVED
          const displayNum = isAdded
            ? line.destination
            : isRemoved
              ? (line.source ?? line.destination)
              : (line.destination ?? line.source);
          lines.push(`${prefix}${displayNum}: ${line.line}`);
        }
      }
    }
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Embedding-based context retrieval
// ---------------------------------------------------------------------------

export async function makeEmbeddingsFromQueries(queries) {
  return Promise.all(
    queries.map(q => embedText(q.query).then(embedding => ({ ...q, embedding: embedding[0] || [] })))
  );
}

export async function searchOpenSearch(repoName, embeddingVector, topK = TOP_K) {
  const opensearchIndexName = `tftools-repo-code-embeddings-${repoName.toLowerCase().trim()}`;
  try {
    const knnResp = await getOsClient().search({
      index: opensearchIndexName,
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
      const rawValues = knnResp.body.hits.hits.map(h => ({ _id: h._id, score: h._score, source: h._source }));
      return Array.from(new Map(rawValues.map(h => [h.source.checksum, h])).values()).slice(0, 3);
    }
  } catch (err) {
    log.warn(`Error querying OpenSearch index ${opensearchIndexName}: ${err.message}`);
  }
  return [];
}

// ---------------------------------------------------------------------------
// Prompt building helpers
// ---------------------------------------------------------------------------

function buildHunkSummaryPrompt(bundle) {
  const sb = [];
  sb.push(`Filepath: ${bundle.filepath}`);
  sb.push(`Changed lines: ${bundle.start}-${bundle.end}`);
  sb.push(`Changed code: \n${bundle.changes}\n`);
  sb.push(`Contextual retrievals (top ${bundle.results.length})`);
  for (const r of bundle.results) {
    sb.push(`--- ${r.source.filepath} ${r.source.start_line || ""}-${r.source.end_line || ""}\n${r.source.content}`);
  }
  sb.push('\nTask: Summarize the purpose of the change and list potential problems, edge cases, or missing considerations. Keep it concise (3-6 bullet points).\n\n');
  return sb.join('\n');
}

/**
 * Merges adjacent hunks that are within gapTolerance lines of each other.
 */
function mergeFileHunks(hunks, gapTolerance = 10) {
  if (!Array.isArray(hunks) || hunks.length === 0) return [];

  const sortedHunks = [...hunks].sort((left, right) => left.start - right.start);
  const merged = [];
  let current = {
    start: sortedHunks[0].start,
    end: sortedHunks[0].end,
    code: Array.isArray(sortedHunks[0].code) ? [...sortedHunks[0].code] : []
  };

  for (let index = 1; index < sortedHunks.length; index++) {
    const next = sortedHunks[index];
    if (next.start <= current.end + gapTolerance) {
      current.end = Math.max(current.end, next.end);
      current.code.push(...(Array.isArray(next.code) ? next.code : []));
      continue;
    }
    merged.push(current);
    current = {
      start: next.start,
      end: next.end,
      code: Array.isArray(next.code) ? [...next.code] : []
    };
  }
  merged.push(current);

  return merged.map(hunk => ({ ...hunk, code: hunk.code.filter(Boolean) }));
}

function buildOpenIssuesContextForPrompt(openIssues = []) {
  if (!Array.isArray(openIssues) || openIssues.length === 0) {
    return "\nPreviously open issues for this file: none.\n";
  }

  const lines = openIssues.flatMap((issue, index) => {
    const locations = Array.isArray(issue.locations) && issue.locations.length > 0
      ? issue.locations.map(location => `${location.filename}:${location.line_number}`).join(", ")
      : `${issue.filename}:${issue.line_number}`;

    return [
      `${index + 1}. ${getSeverityIcon(issue.severity)} ${issue.issue_description}`,
      `Locations: ${locations}`,
      `Suggested fix: ${(issue?.suggestion?.text || "").trim()}`
    ];
  });

  return [
    "",
    "Previously open issues for this file:",
    ...lines,
    "",
    "Important: explicitly verify whether each previously open issue is still present or has been resolved by the latest changes.",
    "If an old issue is fixed, do not report it again as an open issue.",
    ""
  ].join("\n");
}

async function buildArchitectureReviewPrompt(prMeta, diff) {
  const preparedDiff = Array.from(diff.values())
    .flatMap(entry => entry.diffs ? entry.diffs : []);
  const diffObj = { diffs: [...preparedDiff] };
  const transformedDiff = transformBitbucketDiff(diffObj);
  const queries = buildHunkQueries(transformedDiff, diffObj, 20);

  const files = preparedDiff.flatMap(fileDiff => {
    const filepath = fileDiff?.destination?.toString || fileDiff?.source?.toString || "unknown";
    if (IGNORE_FILENAMES.some(name => filepath.endsWith(name))) return [];

    const sourcePath = fileDiff?.source?.toString || null;
    const destinationPath = fileDiff?.destination?.toString || null;
    const isDeleted = Boolean(fileDiff?.deleted) || (!destinationPath && !sourcePath);
    const isNew = Boolean(fileDiff?.new) || (!sourcePath && !destinationPath);
    const isRenameLike = sourcePath && destinationPath && sourcePath !== destinationPath && !isNew && !isDeleted;
    const matchingQueries = queries.filter(q => q.filepath === filepath);

    return [{
      filepath,
      sourcePath,
      destinationPath,
      status: isDeleted ? "deleted" : isNew ? "new" : isRenameLike ? "renamed_or_moved" : "modified",
      snippet: matchingQueries.map(q => q.snippet)
    }];
  });

  return { prompt: getArchitectureReviewPrompt(prMeta, files), files };
}

export async function buildFileReviewPrompt(prMeta, filepath, fileDiff, existingOpenIssues = []) {
  const singleFileDiff = { diffs: [...fileDiff] };
  const transformedDiff = transformBitbucketDiff(singleFileDiff);
  log.info(`transformedDiff for "${filepath}": ${JSON.stringify(transformedDiff)}`);
  const fileEntry = transformedDiff.find(f => f.filepath === filepath);
  if (!fileEntry || !fileEntry.hunks.length) return null;

  const mergedFileEntry = { ...fileEntry, hunks: mergeFileHunks(fileEntry.hunks, 20) };
  const queries = buildHunkQueries([mergedFileEntry], singleFileDiff);
  const embeddings = await makeEmbeddingsFromQueries(queries);
  const hunkSummaries = [];

  for (const q of embeddings) {
    const searchResults = await searchOpenSearch(prMeta.repo, q.embedding, 5);
    const results = (searchResults || []).map(r => {
      const { embedding, ...filteredQ } = r.source;
      return { id: r._id, score: r.score, source: filteredQ };
    });

    const { embedding, ...filteredQ } = q;
    hunkSummaries.push({
      filename: filteredQ.filepath,
      hunk: { start: filteredQ.start, end: filteredQ.end },
      summary: buildHunkSummaryPrompt({ ...filteredQ, results })
    });
  }

  const basePrompt = getFileReviewPrompt(prMeta, singleFileDiff, filepath, hunkSummaries.map(h => h.summary));
  const openIssuesContext = buildOpenIssuesContextForPrompt(existingOpenIssues);

  return { filepath, prompt: `${basePrompt}\n${openIssuesContext}` };
}

export async function buildPerFileReviewPrompts(prMeta, changedFilesDiffs, openIssues = []) {
  const prompts = [];
  for (const changedFilename of changedFilesDiffs.keys()) {
    if (IGNORE_FILENAMES.some(name => changedFilename.endsWith(name))) continue;

    log.info(`Building file review prompt for "${changedFilename}"`);
    const filePrompt = await buildFileReviewPrompt(
      prMeta,
      changedFilename,
      changedFilesDiffs.get(changedFilename)?.diffs || [],
      openIssues.filter(openIssue => openIssue.locations.some(loc => loc.filename === changedFilename))
    );

    if (filePrompt) {
      prompts.push({ ...filePrompt, status: "OPEN" });
    }
  }
  return prompts;
}

// ---------------------------------------------------------------------------
// Issue normalization helpers
// ---------------------------------------------------------------------------

export function normalizeIssueDescription(text = "") {
  return String(text).toLowerCase().replace(/`+/g, "").replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
}

export function normalizeSeverity(severity) {
  const normalized = String(severity || "MEDIUM").trim().toUpperCase();
  return Object.prototype.hasOwnProperty.call(ISSUE_SEVERITY_ORDER, normalized) ? normalized : "MEDIUM";
}

export function getSeverityIcon(severity) {
  switch (normalizeSeverity(severity)) {
    case "CRITICAL": return "🔴";
    case "HIGH":     return "🟠";
    case "MEDIUM":   return "🟡";
    case "LOW":      return "🔵";
    default:         return "⚪";
  }
}

export function formatIssueLocations(issue) {
  const locations = Array.isArray(issue?.locations) ? issue.locations : [];

  if (locations.length === 0) {
    const fallbackFilename = typeof issue?.filename === "string" ? issue.filename.trim() : "";
    const fallbackLine = typeof issue?.line_number === "number"
      ? (Number.isFinite(Number(issue?.line_number)) ? Number(issue.line_number) : null)
      : null;

    return fallbackFilename && fallbackLine !== null
      ? `${fallbackFilename}:${fallbackLine}`
      : "Location unavailable";
  }

  return locations
    .map(location => {
      const filename = typeof location?.filename === "string" ? location.filename.trim() : "";
      const lineNumber = typeof location?.line_number === "number"
        ? (Number.isFinite(Number(location?.line_number)) ? Number(location.line_number) : null)
        : null;
      // REMOVED lines carry a source (FROM-side) line number; all others are destination (TO-side)
      const sideParam = location?.line_type === "REMOVED" ? "f" : "t";

      return filename && lineNumber !== null
        ? `[${filename}:${lineNumber}](diff#${encodeURIComponent(filename)}?${sideParam}=${lineNumber})`
        : null;
    })
    .filter(Boolean)
    .join(", ");
}

export function getIssueLineNumber(issue) {
  return typeof issue?.line_number === "number" ? issue.line_number : null;
}

export function buildIssueKey(issue) {
  return [
    issue?.filename || "",
    normalizeIssueDescription(issue?.issue_description || ""),
    String(getIssueLineNumber(issue))
  ].join("|");
}

export function areIssuesEquivalent(leftIssue, rightIssue) {
  if (!leftIssue || !rightIssue) return false;
  if ((leftIssue.filename || "") !== (rightIssue.filename || "")) return false;
  const leftLine = getIssueLineNumber(leftIssue);
  const rightLine = getIssueLineNumber(rightIssue);
  if (leftLine !== null && rightLine !== null && Math.abs(leftLine - rightLine) > IGNORE_LINE_TOLERANCE) {
    return false;
  }
  return normalizeIssueDescription(leftIssue.issue_description) === normalizeIssueDescription(rightIssue.issue_description);
}

export function getPrimaryLocation(issue) {
  const locations = Array.isArray(issue?.locations) ? issue.locations : [];

  for (const location of locations) {
    const filename = typeof location?.filename === "string" ? location.filename.trim() : "";
    const lineNumber = typeof location?.line_number === "number"
      ? location.line_number
      : (Number.isFinite(Number(location?.line_number)) ? Number(location.line_number) : null);

    if (filename && lineNumber !== null) return { filename, line_number: lineNumber };
  }

  const fallbackFilename = typeof issue?.filename === "string" ? issue.filename.trim() : "";
  const fallbackLineNumber = typeof issue?.line_number === "number"
    ? issue.line_number
    : (Number.isFinite(Number(issue?.line_number)) ? Number(issue.line_number) : null);

  if (fallbackFilename && fallbackLineNumber !== null) {
    return { filename: fallbackFilename, line_number: fallbackLineNumber };
  }

  return null;
}

function sortIssuesBySeverity(issues) {
  return [...issues].sort((left, right) => {
    const leftRank = ISSUE_SEVERITY_ORDER[normalizeSeverity(left?.severity)];
    const rightRank = ISSUE_SEVERITY_ORDER[normalizeSeverity(right?.severity)];
    if (leftRank !== rightRank) return leftRank - rightRank;
    return buildIssueKey(left).localeCompare(buildIssueKey(right));
  });
}

export function normalizeReviewIssues(review) {
  if (!review) return [];

  let parsedReview = review;
  if (typeof parsedReview === "string") {
    try {
      parsedReview = JSON.parse(parsedReview);
    } catch (error) {
      log.warn(`Failed to parse review payload while normalizing issues: ${error}`);
      return [];
    }
  }

  const potentialIssues = Array.isArray(parsedReview?.potential_issues) ? parsedReview.potential_issues : [];

  return potentialIssues.flatMap(issue => {
    const issueDescription = typeof issue?.issue_description === "string" ? issue.issue_description.trim() : "";
    const suggestion = typeof issue?.suggestion === "object" ? issue.suggestion : {};
    const severity = normalizeSeverity(issue?.severity);
    const locations = Array.isArray(issue?.locations) ? issue.locations : [];

    if (!issueDescription || locations.length === 0) return [];

    return locations.map(location => {
      const filename = typeof location?.filename === "string" ? location.filename.trim() : "";
      const lineNumber = typeof location?.line_number === "number"
        ? (Number.isFinite(Number(location.line_number)) ? Number(location.line_number) : null)
        : null;
      const lineType = ["ADDED", "REMOVED", "CONTEXT"].includes(location?.line_type)
        ? location.line_type
        : null;

      if (!filename || lineNumber === null) return null;

      return {
        filename,
        line_number: lineNumber,
        line_type: lineType,
        issue_description: issueDescription,
        severity,
        suggestion,
        locations: [{ filename, line_number: lineNumber, line_type: lineType }]
      };
    }).filter(Boolean);
  });
}

export function collectLatestIssuesFromFileReviews(fileReviews) {
  return fileReviews.flatMap(fileReview => normalizeReviewIssues(fileReview.review));
}

export function buildNormalizedReviewFromIssues(issues) {
  const grouped = new Map();

  for (const issue of issues) {
    const key = [
      normalizeIssueDescription(issue.issue_description),
      normalizeSeverity(issue.severity),
      JSON.stringify(issue.suggestion || {})
    ].join("::");

    if (!grouped.has(key)) {
      grouped.set(key, {
        issue_description: issue.issue_description,
        severity: normalizeSeverity(issue.severity),
        suggestion: issue.suggestion || {},
        locations: []
      });
    }

    const group = grouped.get(key);
    for (const location of issue.locations || []) {
      if (!group.locations.some(ex => ex.filename === location.filename && ex.line_number === location.line_number)) {
        group.locations.push({ filename: location.filename, line_number: location.line_number });
      }
    }
  }

  return {
    potential_issues: Array.from(grouped.values()).map(issue => ({
      ...issue,
      locations: issue.locations.sort((l, r) => l.filename.localeCompare(r.filename) || l.line_number - r.line_number)
    }))
  };
}

export async function normalizeIssuesWithLLM(prMeta, issues) {
  if (!Array.isArray(issues) || issues.length === 0) {
    log.info(`No issues to normalize for PR ${prMeta.pr} in repo ${prMeta.repo}`);
    return { normalizedIssues: [], resolvedIssues: [] };
  }

  const normalizationPrompt = getIssuesNormalizationPrompt(prMeta, buildNormalizedReviewFromIssues(issues).potential_issues);
  const normalizedResponse = await queryLLM(normalizationPrompt, {
    generation_config: {
      responseMimeType: "application/json",
      responseSchema: pr_review_issues_schema
    }
  });

  const normalizedIssues = sortIssuesBySeverity(normalizeReviewIssues(normalizedResponse));
  return {
    normalizedIssues: normalizedIssues.slice(0, MAX_POSTED_ISSUES),
    resolvedIssues: normalizedIssues.slice(MAX_POSTED_ISSUES).map(issue => ({ ...issue, status: "RESOLVED" }))
  };
}

export function buildIssuesReviewFromList(issues) {
  const normalizedIssues = Array.isArray(issues)
    ? issues.map(issue => {
        const issueDescription = typeof issue?.issue_description === "string" ? issue.issue_description.trim() : "";
        const suggestion = typeof issue?.suggestion === "object" ? issue.suggestion : {};
        const severity = normalizeSeverity(issue?.severity);
        const locations = Array.isArray(issue?.locations)
          ? issue.locations.map(location => {
              const filename = typeof location?.filename === "string" ? location.filename.trim() : "";
              const lineNumber = typeof location?.line_number === "number"
                ? location.line_number
                : (Number.isFinite(Number(location?.line_number)) ? Number(location.line_number) : null);
              if (!filename || lineNumber === null) return null;
              return { filename, line_number: lineNumber };
            }).filter(Boolean)
          : [];

        if (!issueDescription || locations.length === 0) return null;

        return { issue_description: issueDescription, suggestion, severity, locations };
      }).filter(Boolean)
    : [];

  return { potential_issues: normalizedIssues };
}

// ---------------------------------------------------------------------------
// Issue lifecycle tracking
// ---------------------------------------------------------------------------

/**
 * Compares previousIssues (Map of {issueKey -> entry}) with latestIssues (array)
 * to produce a status array of {issue, status: "NEW"|"OPEN"|"RESOLVED"}.
 */
export function buildIssueStatusesForSummary({ previousIssues, latestIssues }) {
  const previousList = Array.isArray(previousIssues)
    ? previousIssues
    : Array.from(previousIssues instanceof Map ? previousIssues.values() : []);
  const latestList = Array.isArray(latestIssues) ? latestIssues : [];
  const issueStatuses = [];
  const matchedPreviousIndexes = new Set();

  for (const latestIssue of latestList) {
    const matchedIndex = previousList.findIndex((entry, index) => {
      if (matchedPreviousIndexes.has(index)) return false;
      return areIssuesEquivalent(entry?.issue, latestIssue);
    });

    if (matchedIndex >= 0) {
      matchedPreviousIndexes.add(matchedIndex);
      issueStatuses.push({ issue: latestIssue, status: "OPEN" });
    } else {
      issueStatuses.push({ issue: latestIssue, status: "NEW" });
    }
  }

  previousList.forEach((entry, index) => {
    if (matchedPreviousIndexes.has(index)) return;
    issueStatuses.push({ issue: entry.issue, status: "RESOLVED" });
  });

  return issueStatuses;
}

export function buildIssueLifecycle({ previousIssues, latestIssues }) {
  log.info(`Building issue lifecycle. Previous: ${previousIssues instanceof Map ? previousIssues.size : Array.isArray(previousIssues) ? previousIssues.length : 0}, Latest: ${Array.isArray(latestIssues) ? latestIssues.length : 0}`);
  const issueStatuses = buildIssueStatusesForSummary({ previousIssues, latestIssues });

  return {
    issueStatuses,
    newIssues: issueStatuses.filter(({ status }) => status === "NEW").map(({ issue }) => issue),
    openIssues: issueStatuses.filter(({ status }) => status === "OPEN").map(({ issue }) => issue),
    resolvedIssues: issueStatuses.filter(({ status }) => status === "RESOLVED").map(({ issue }) => issue)
  };
}

// ---------------------------------------------------------------------------
// Summary comment text building
// ---------------------------------------------------------------------------

export function buildSummaryCommentText(reviewParsed, issueStatuses) {
  const statusByIssueKey = new Map(
    (issueStatuses || []).map(({ issue, status }) => [buildIssueKey(issue), status])
  );

  const potentialIssues = Array.isArray(reviewParsed?.potential_issues) ? reviewParsed.potential_issues : [];

  const suggestionsBlock = potentialIssues.length > 0
    ? potentialIssues.map(issue => {
        const status = statusByIssueKey.get(buildIssueKey(issue)) || "OPEN";
        switch (status) {
          case "OPEN":
          case "NEW":
            return [
              `* ${getSeverityIcon(issue.severity)} ${issue.issue_description}`,
              `**Location(s):** ${formatIssueLocations(issue)}`,
              `**Suggestion:** ${issue?.suggestion?.text || ""}`
            ].join("\n");
          case "RESOLVED":
            return [
              `* ✅ ${getSeverityIcon(issue.severity)} ~~${issue.issue_description}~~`,
              `~~**Location(s):** ${formatIssueLocations(issue)}~~`
            ].join("\n");
          default:
            return "";
        }
      }).join("\n\n")
    : "No open issues in the latest review iteration.";

  return [
    "### Review Summary",
    "",
    reviewParsed.summary || "",
    "",
    "### Potential Issues",
    "",
    suggestionsBlock,
    "",
    "### Verdict",
    "",
    reviewParsed.verdict || "No clear verdict provided."
  ].join("\n");
}

export function buildAggregateSummaryPrompt(prMeta, fileReviews) {
  return getAggregateSummaryPrompt(prMeta, fileReviews);
}

// ---------------------------------------------------------------------------
// LLM review runners
// ---------------------------------------------------------------------------

export async function runArchitecturalReview(prMeta, changedFilesDiffs, prId) {
  const architecturalPrompt = await buildArchitectureReviewPrompt(prMeta, changedFilesDiffs);
  log.info(`Built architectural review prompt for PR #${prId} with ${architecturalPrompt.files.length} files.`);
  log.info(`Sending architectural review prompt to LLM for PR #${prId} (${architecturalPrompt.prompt.length} chars)...`);

  const architecturalReview = await queryLLM(architecturalPrompt.prompt, {
    generation_config: {
      responseMimeType: "application/json",
      responseSchema: architecture_review_schema
    }
  });
  log.info(`Received architectural review response for PR #${prId}: ${JSON.stringify(architecturalReview)}`);

  try {
    const parsed = typeof architecturalReview === "string" ? JSON.parse(architecturalReview) : architecturalReview;
    if (!parsed || !Array.isArray(parsed.review_buckets)) {
      return { parsed: { summary: "Architectural review failed.", review_buckets: [] }, preface: "" };
    }
    return { parsed, preface: getArchitectureReviewPreface(parsed.summary) };
  } catch (error) {
    log.error(`Failed to parse architectural review for PR #${prId}: ${error}`);
    return { parsed: { summary: "Architectural review failed.", review_buckets: [] }, preface: "" };
  }
}

export async function runBucketReviews({ reviewBuckets, filePrompts, architecturalReviewPromptPreface }) {
  const fileReviews = [];
  const consumedFilePaths = new Set();

  for (const reviewBucket of reviewBuckets) {
    const bucketFilePrompts = filePrompts.filter(
      fp => reviewBucket.files.includes(fp.filepath) && !consumedFilePaths.has(fp.filepath)
    );
    if (bucketFilePrompts.length === 0) continue;

    const bucketReviewPrompt = architecturalReviewPromptPreface +
      getBucketReviewPreface(reviewBucket) +
      bucketFilePrompts.map(fp => fp.prompt + "\n========== NEXT_FILE ==========\n").join("\n");

    log.info(`Bucket review: [${reviewBucket.files.join(", ")}] (${bucketReviewPrompt.length} chars)`);
    const bucketReview = await queryLLM(bucketReviewPrompt, {
      generation_config: {
        responseMimeType: "application/json",
        responseSchema: pr_review_issues_schema
      }
    });
    const bucketReviewParsed = typeof bucketReview === "string" ? JSON.parse(bucketReview) : bucketReview;

    fileReviews.push({ type: "FILE_BUCKET_REVIEW", filepath: reviewBucket.files.join(","), review: bucketReviewParsed });
    for (const fp of bucketFilePrompts) consumedFilePaths.add(fp.filepath);
  }

  return { fileReviews, consumedFilePaths };
}

export async function runSingleFileReviews({ filePrompts, consumedFilePaths, architecturalReviewPromptPreface }) {
  const fileReviews = [];

  for (const filePrompt of filePrompts.filter(fp => !consumedFilePaths.has(fp.filepath))) {
    const reviewFilePrompt = architecturalReviewPromptPreface + "\n" + filePrompt.prompt;
    log.info(`Single-file review: ${filePrompt.filepath} (${reviewFilePrompt.length} chars)`);
    const review = await queryLLM(reviewFilePrompt, {
      generation_config: {
        responseMimeType: "application/json",
        responseSchema: pr_review_issues_schema
      }
    });

    const reviewParsed = typeof review === "string" ? JSON.parse(review) : review;
    fileReviews.push({ type: "FILE_SINGLE_REVIEW", filepath: filePrompt.filepath, review: reviewParsed });
  }

  return { fileReviews };
}

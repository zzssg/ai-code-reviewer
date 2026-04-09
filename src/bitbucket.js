/**
 * Bitbucket Comment Management Service.
 * Responsible for:
 *  - Loading PR context (metadata, diffs, changed files) from the Bitbucket API
 *  - Syncing review comments (inline + summary) with Bitbucket
 *
 * This module reads review state from OpenSearch (via review-engine)
 * and makes all HTTP calls to the Bitbucket REST API.
 */

import * as dotenv from 'dotenv';
dotenv.config();

import axios from "axios";
import { createLogger } from "./utils.js";
import { DIFF_SEGMENT_TYPES } from "./types.js";
import { loadReviewState } from "./review-engine.js";

const IGNORE_LINE_TOLERANCE = Number(process.env.PR_REVIEWER_ISSUE_LINE_TOLERANCE || 5);

const log = createLogger(import.meta.url);

const BITBUCKET_TOKEN = process.env.BITBUCKET_TOKEN;
const BITBUCKET_BASE = process.env.BITBUCKET_BASE || "https://bb/rest/api/1.0";
const BITBUCKET_HEADERS = { Authorization: `Bearer ${BITBUCKET_TOKEN}` };
const IGNORE_FILENAMES = ["package-lock.json", "package.json", "gradle.properties"];

// ---------------------------------------------------------------------------
// PR context loading (read-only Bitbucket calls)
// ---------------------------------------------------------------------------

/**
 * Loads all PR context needed for the review: metadata, full diff, and per-file
 * diffs since the last processed commit.
 *
 * @param {object} params
 * @param {object} params.event - Bitbucket webhook payload
 * @returns {object} prId, prDetailsUrl, postCommentUrl, diff, prMeta, reviewId,
 *                   reviewState, changedFilesDiffs, prDetails
 */
export async function loadPrContext({ event }) {
  const pr = event.pullRequest;
  const prId = pr.id;
  const sourceRepo = pr.fromRef?.repository;
  const destRepo = pr.toRef?.repository || pr.destination?.repository;
  const repoSlug = sourceRepo?.slug || destRepo?.slug;
  const projectKey = sourceRepo?.project?.key || destRepo?.project?.key;

  if (!repoSlug || !projectKey) {
    throw new Error("Missing repository information in webhook payload");
  }

  const baseUrl = `${BITBUCKET_BASE}/projects/${encodeURIComponent(projectKey)}/repos/${encodeURIComponent(repoSlug)}`;
  const prDetailsUrl = `${baseUrl}/pull-requests/${prId}`;
  const prChangesUrl = `${prDetailsUrl}/changes`;
  const diffUrl = `${prDetailsUrl}/diff`;
  const commitsUrl = `${baseUrl}/commits`;
  const postCommentUrl = `${prDetailsUrl}/comments`;

  const [prDetailsRes, diffRes] = await Promise.all([
    axios.get(prDetailsUrl, { headers: BITBUCKET_HEADERS }),
    axios.get(diffUrl, { headers: BITBUCKET_HEADERS })
  ]);
  const prDetails = prDetailsRes.data;
  const diff = diffRes.data;

  const prMeta = {
    project: projectKey,
    repo: repoSlug,
    pr: prId,
    from: diff.fromHash,
    to: diff.toHash
  };

  const reviewId = `${repoSlug}_${prId}`;
  const reviewState = await loadReviewState(reviewId);
  const lastProcessedCommitId = reviewState?.toHash || prMeta.from;
  const latestCommitId = prMeta.to;

  log.info(`PR #${prId}: last processed commit=${lastProcessedCommitId}, latest commit=${latestCommitId}`);

  const prChangesRes = await axios.get(
    `${prChangesUrl}?changeScope=RANGE&sinceId=${lastProcessedCommitId}&untilId=${latestCommitId}`,
    { headers: BITBUCKET_HEADERS }
  );
  const prChanges = prChangesRes.data;
  const listOfChangedFiles = (prChanges.values || [])
    .map(value => value.path?.toString)
    .filter(Boolean)
    .filter(fp => !IGNORE_FILENAMES.some(name => fp.endsWith(name)));

  log.info(`PR #${prId} changed files since ${lastProcessedCommitId}: ${listOfChangedFiles.join(", ")}`);

  const changedFilesDiffs = new Map();
  await Promise.all(listOfChangedFiles.map(async filePath => {
    const fileDiffUrl = `${commitsUrl}/${latestCommitId}/diff/${filePath}?contextLines=10&sinceId=${lastProcessedCommitId}&whitespace=show&withComments=false`;
    const fileDiffRes = await axios.get(fileDiffUrl, { headers: BITBUCKET_HEADERS });
    changedFilesDiffs.set(
      filePath,
      typeof fileDiffRes.data === "string" ? JSON.parse(fileDiffRes.data) : fileDiffRes.data
    );
  }));

  return {
    prId,
    prDetailsUrl,
    prChangesUrl,
    postCommentUrl,
    prMeta,
    diff,
    changedFilesDiffs,
    reviewId,
    reviewState,
    prDetails
  };
}

// ---------------------------------------------------------------------------
// Anchor resolution helpers (for inline comment placement)
// ---------------------------------------------------------------------------

function findFileDiffByPath(diff, filepath) {
  return (diff?.diffs || []).find(fileDiff =>
    fileDiff?.destination?.toString === filepath ||
    fileDiff?.source?.toString === filepath
  ) || null;
}

function buildOriginalLineMap(fileDiff) {
  const map = new Map();
  for (const hunk of fileDiff?.hunks || []) {
    for (const segment of hunk.segments || []) {
      for (const line of segment.lines || []) {
        const sourceLine = typeof line.source === "number" ? line.source : null;
        const destinationLine = typeof line.destination === "number" ? line.destination : null;
        if (sourceLine !== null) {
          map.set(sourceLine, destinationLine ?? sourceLine);
        }
      }
    }
  }
  return map;
}

/**
 * Finds the closest line to requestedLine within the tolerance range.
 * Returns { line, distance, segmentType, sourceLine, destinationLine } or null.
 */
function findClosestLineInTolerance(fileDiff, requestedLine, tolerance = IGNORE_LINE_TOLERANCE) {
  let closestMatch = null;
  let minDistance = tolerance + 1;

  for (const hunk of fileDiff?.hunks || []) {
    for (const seg of hunk.segments || []) {
      for (const line of seg.lines || []) {
        const sourceLine = typeof line.source === "number" ? line.source : null;
        const destinationLine = typeof line.destination === "number" ? line.destination : null;

        // Check destination line (for ADDED and CONTEXT on TO side)
        if (destinationLine !== null) {
          const distance = Math.abs(destinationLine - requestedLine);
          if (distance <= tolerance && distance < minDistance) {
            minDistance = distance;
            closestMatch = {
              line: destinationLine,
              distance,
              segmentType: seg.type,
              sourceLine,
              destinationLine,
              isDestination: true
            };
          }
        }

        // Check source line (for REMOVED and CONTEXT on FROM side)
        if (sourceLine !== null) {
          const distance = Math.abs(sourceLine - requestedLine);
          if (distance <= tolerance && distance < minDistance) {
            minDistance = distance;
            closestMatch = {
              line: sourceLine,
              distance,
              segmentType: seg.type,
              sourceLine,
              destinationLine,
              isDestination: false
            };
          }
        }
      }
    }
  }

  return closestMatch;
}

function findAnchorInfoForIssue(fileDiff, issue) {
  const requestedLine = issue?.line_number;
  if (!fileDiff || typeof requestedLine !== "number") return null;

  let bestContextMatch = null;
  const originalLineMap = buildOriginalLineMap(fileDiff);

  // PHASE 1: Try exact matches
  for (const hunk of fileDiff.hunks || []) {
    for (const seg of hunk.segments || []) {
      const segmentType = seg.type;
      for (const line of seg.lines || []) {
        const sourceLine = typeof line.source === "number" ? line.source : null;
        const destinationLine = typeof line.destination === "number" ? line.destination : null;

        if (segmentType === DIFF_SEGMENT_TYPES.ADDED && destinationLine === requestedLine) {
          log.debug(`Exact match: ADDED line ${destinationLine} for issue line ${requestedLine}`);
          return {
            line: destinationLine,
            lineType: "ADDED",
            fileType: "TO",
            path: fileDiff?.destination?.toString || issue.filename
          };
        }

        if (segmentType === DIFF_SEGMENT_TYPES.REMOVED && sourceLine === requestedLine) {
          log.debug(`Exact match: REMOVED line ${sourceLine} for issue line ${requestedLine}`);
          return {
            line: sourceLine,
            lineType: "REMOVED",
            fileType: "FROM",
            path: fileDiff?.source?.toString || issue.filename
          };
        }

        if (segmentType === DIFF_SEGMENT_TYPES.CONTEXT) {
          if (destinationLine === requestedLine) {
            bestContextMatch = {
              line: destinationLine,
              lineType: "CONTEXT",
              fileType: "TO",
              path: fileDiff?.destination?.toString || issue.filename
            };
          }
          if (sourceLine === requestedLine && !bestContextMatch) {
            bestContextMatch = {
              line: sourceLine,
              lineType: "CONTEXT",
              fileType: "FROM",
              path: fileDiff?.source?.toString || issue.filename
            };
          }
        }
      }
    }
  }

  if (bestContextMatch) {
    log.debug(`Exact match: CONTEXT line ${bestContextMatch.line} for issue line ${requestedLine}`);
    return bestContextMatch;
  }

  // PHASE 2: Try fuzzy matching within tolerance if no exact match found
  const fuzzyMatch = findClosestLineInTolerance(fileDiff, requestedLine);
  if (fuzzyMatch) {
    log.warn(
      `Fuzzy match: Line numbers differ by ${fuzzyMatch.distance} lines. ` +
      `Requested ${requestedLine}, using closest ${fuzzyMatch.segmentType} line ${fuzzyMatch.line} ` +
      `for ${issue.filename}`
    );

    // Determine anchor based on segment type and which line was closest
    if (fuzzyMatch.segmentType === DIFF_SEGMENT_TYPES.ADDED && fuzzyMatch.isDestination) {
      return {
        line: fuzzyMatch.line,
        lineType: "ADDED",
        fileType: "TO",
        path: fileDiff?.destination?.toString || issue.filename
      };
    }

    if (fuzzyMatch.segmentType === DIFF_SEGMENT_TYPES.REMOVED && !fuzzyMatch.isDestination) {
      return {
        line: fuzzyMatch.line,
        lineType: "REMOVED",
        fileType: "FROM",
        path: fileDiff?.source?.toString || issue.filename
      };
    }

    if (fuzzyMatch.segmentType === DIFF_SEGMENT_TYPES.CONTEXT) {
      return {
        line: fuzzyMatch.line,
        lineType: "CONTEXT",
        fileType: fuzzyMatch.isDestination ? "TO" : "FROM",
        path: fuzzyMatch.isDestination
          ? (fileDiff?.destination?.toString || issue.filename)
          : (fileDiff?.source?.toString || issue.filename)
      };
    }
  }

  // PHASE 3: Fallback to originalLineMap lookup
  if (originalLineMap.has(requestedLine)) {
    log.debug(`Fallback: Using originalLineMap for line ${requestedLine}`);
    return {
      line: requestedLine,
      lineType: "CONTEXT",
      fileType: "FROM",
      path: fileDiff?.source?.toString || issue.filename
    };
  }

  // PHASE 4: Last resort - default to added line on TO side
  log.warn(
    `No match found (exact, fuzzy, or in map) for ${issue.filename}:${requestedLine}. ` +
    `Defaulting to ADDED line on TO side. This comment may be misplaced.`
  );
  return {
    line: requestedLine,
    lineType: "ADDED",
    fileType: "TO",
    path: fileDiff?.destination?.toString || issue.filename
  };
}

// ---------------------------------------------------------------------------
// Comment posting / management
// ---------------------------------------------------------------------------

/**
 * Posts a single inline comment for a review issue.
 * Returns the posted comment data (or {} in dry-run mode).
 */
export async function postInlineComment(url, issue, diff) {
  if (!issue?.filename || typeof issue.line_number !== "number") {
    log.warn(`Bad input for inline comment — missing filename or line_number: ${JSON.stringify(issue)}`);
    return null;
  }

  const fileDiff = findFileDiffByPath(diff, issue.filename);
  const anchorInfo = findAnchorInfoForIssue(fileDiff, issue);
  if (!anchorInfo) {
    log.warn(`Unable to resolve inline anchor for ${issue.filename}:${issue.line_number}`);
    return null;
  }

  const textParts = [issue.issue_description];
  if (issue.suggestion) {
    textParts.push("Suggested fix:\r\n");
    if (issue.suggestion?.text) textParts.push(issue.suggestion.text);
    if (issue.suggestion?.code) textParts.push(issue.suggestion.code);
  }

  const payload = {
    text: textParts.join("\r\n"),
    anchor: {
      line: anchorInfo.line,
      lineType: anchorInfo.lineType,
      fileType: anchorInfo.fileType,
      path: anchorInfo.path
    }
  };

  if (process.env.PR_REVIEWER_DRYRUN === "true") {
    log.info(`[DRY RUN] Would post inline comment: ${JSON.stringify(payload)}`);
    return {};
  }

  const response = await axios.post(url, payload, { headers: BITBUCKET_HEADERS });
  return response.data;
}

/**
 * Posts inline comments for all issues in a review, optionally filtering by filepath.
 */
export async function postInlineCommentsForReview({ url, filepath = "", review, diff } = {}) {
  const { normalizeReviewIssues } = await import("./review-engine.js");
  let issues = normalizeReviewIssues(review);
  if (filepath) {
    issues = issues.filter(issue => issue.filename === filepath);
  }

  const postedComments = [];
  for (const issue of issues) {
    try {
      const postedComment = await postInlineComment(url, issue, diff);
      postedComments.push({ issue, postedComment });
    } catch (err) {
      log.error(`Failed to post inline comment for ${issue.filename}:${issue.line_number}: ${err.message}`);
      postedComments.push({ issue, postedComment: null });
    }
  }

  return postedComments;
}

/**
 * Creates a new comment or updates an existing one (PUT with version for optimistic locking).
 */
export async function upsertReviewComment({ url, existingComment, text = "", flags = {} }) {
  const commentPayload = text.trim().length > 0 ? { text, ...flags } : { ...flags };

  if (existingComment?.id) {
    const updateUrl = `${url}/${existingComment.id}`;
    const version = existingComment.version || 0;
    log.info(`Updating comment ${existingComment.id} with payload: ${JSON.stringify({ version, ...commentPayload })}`);
    const response = await axios.put(updateUrl, { version, ...commentPayload }, { headers: BITBUCKET_HEADERS });
    return response.data;
  }

  const response = await axios.post(url, commentPayload, { headers: BITBUCKET_HEADERS });
  return response.data;
}

/**
 * Marks a comment thread as resolved.
 */
export async function resolveReviewComment({ url, existingComment }) {
  return upsertReviewComment({
    url,
    existingComment,
    flags: { threadResolved: true }
  });
}

/**
 * Deletes a comment by ID.
 */
export async function deleteReviewComment({ url, commentId }) {
  if (!commentId) return null;
  const deleteUrl = `${url}/${commentId}`;
  log.info(`Deleting inline comment ${commentId}`);
  const response = await axios.delete(deleteUrl, { headers: BITBUCKET_HEADERS });
  return response.data;
}

/**
 * Resolves (or deletes) all inline comments for issues that have been resolved
 * in the latest review pass.
 *
 * @param {object} params
 * @param {string} params.url - Bitbucket comments URL
 * @param {Map}    params.previousIssues - Map of issueKey -> {id, issue, ...}
 * @param {Array}  params.issueStatuses - Array of {issue, status} from buildIssueLifecycle
 * @param {boolean} params.deleteResolved - Delete comment instead of resolving thread
 */
export async function resolveResolvedComments({ url, previousIssues, issueStatuses, deleteResolved = false }) {
  const { areIssuesEquivalent } = await import("./review-engine.js");

  for (const { issue, status } of issueStatuses) {
    if (status !== "RESOLVED") continue;

    const previousEntry = Array.from(previousIssues.values()).find(entry =>
      areIssuesEquivalent(entry.issue, issue)
    );

    if (!previousEntry?.id) continue;

    try {
      if (deleteResolved) {
        await deleteReviewComment({ url, commentId: previousEntry.id });
        log.info(`Deleted inline comment ${previousEntry.id} for ${issue.filename}:${issue.line_number}`);
      } else {
        await resolveReviewComment({ url, existingComment: previousEntry });
        log.info(`Resolved inline comment ${previousEntry.id} for ${issue.filename}:${issue.line_number}`);
      }
    } catch (error) {
      log.warn(`Failed to manage inline comment ${previousEntry.id}: ${error.message}`);
    }
  }
}

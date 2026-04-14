/**
 * Prompt construction functions for the AI code review system.
 * Each function returns a prompt string to be sent to the LLM.
 */

/**
 * Builds the architectural review prompt for the LLM.
 * @param {object} prMeta - PR metadata (project, repo, pr, from, to)
 * @param {Array} files - Array of file objects with filepath, status, snippet
 */
export function getArchitectureReviewPrompt(prMeta, files) {
  const fileList = (files || []).map(f => {
    const statusLabel = f.status === "deleted" ? " [DELETED]"
      : f.status === "new" ? " [NEW]"
      : f.status === "renamed_or_moved" ? " [RENAMED/MOVED]"
      : "";
    const snippets = Array.isArray(f.snippet) && f.snippet.some(Boolean)
      ? f.snippet.filter(Boolean).join("\n---\n")
      : "(no diff available)";
    return `### ${f.filepath}${statusLabel}\n${snippets}`;
  }).join("\n\n");

  return [
    "You are a senior software engineer performing an architectural analysis of a pull request.",
    "",
    "## Pull Request",
    `- Project: ${prMeta.project}`,
    `- Repository: ${prMeta.repo}`,
    `- PR ID: ${prMeta.pr}`,
    "",
    "## Changed Files",
    fileList || "(no changed files)",
    "",
    "## Task",
    "Analyze the structure and intent of these changes:",
    "1. Identify structural movements: renames, deletions, merges, splits, code moves.",
    "2. Group related files into logical review buckets (files that should be reviewed together).",
    "3. Write a concise architectural summary describing what this PR changes at a high level.",
    "",
    "Respond with JSON matching the schema provided."
  ].join("\n");
}

/**
 * Returns a preface string derived from the architectural review summary,
 * used as context prefix for subsequent per-file or per-bucket review prompts.
 * @param {string} summary - Architectural summary from the LLM
 */
export function getArchitectureReviewPreface(summary) {
  if (!summary || typeof summary !== "string" || !summary.trim()) return "";
  return [
    "## Architectural Context",
    "",
    summary.trim(),
    ""
  ].join("\n");
}

/**
 * Returns a preface string for a review bucket, used as context before reviewing
 * the files in that bucket.
 * @param {object} reviewBucket - Bucket with name, goal, rationale, files[]
 */
export function getBucketReviewPreface(reviewBucket) {
  return [
    `## Review Bucket: ${reviewBucket.name}`,
    `**Goal:** ${reviewBucket.goal}`,
    `**Rationale:** ${reviewBucket.rationale}`,
    `**Files in this bucket:** ${(reviewBucket.files || []).join(", ")}`,
    ""
  ].join("\n");
}

/**
 * Builds the per-file review prompt for the LLM.
 * @param {object} prMeta - PR metadata
 * @param {object} singleFileDiff - The diff object for this file
 * @param {string} filepath - The file path being reviewed
 * @param {string[]} hunkSummaries - Array of hunk analysis strings
 */
export function getFileReviewPrompt(prMeta, singleFileDiff, filepath, hunkSummaries) {
  const summariesSection = Array.isArray(hunkSummaries) && hunkSummaries.length > 0
    ? hunkSummaries.join("\n\n---\n\n")
    : "(no hunk analysis available)";

  return [
    "You are a senior software engineer reviewing a pull request.",
    "",
    "## Pull Request",
    `- Project: ${prMeta.project}`,
    `- Repository: ${prMeta.repo}`,
    `- PR ID: ${prMeta.pr}`,
    "",
    `## File Under Review: \`${filepath}\``,
    "",
    "## Change Analysis by Hunk",
    summariesSection,
    "",
    "## Line Number Format",
    "Each line in the hunk analysis is shown with a prefix and its exact line number:",
    "  `+N: <code>` — line N was **added** to the new file (destination). Use N as line_number, line_type=ADDED.",
    "  `-N: <code>` — line N was **removed** from the old file (source). Use N as line_number, line_type=REMOVED.",
    "  ` N: <code>` — line N is unchanged context. Use N as line_number, line_type=CONTEXT.",
    "Always use the exact N from the prefix as line_number. Never guess or approximate.",
    "",
    "## Task",
    "Review the changes in this file. Identify any of the following:",
    "- Logic errors or incorrect assumptions",
    "- Missing or incorrect error handling",
    "- Security vulnerabilities (injection, XSS, auth bypass, etc.)",
    "- Performance issues",
    "- Race conditions or concurrency bugs",
    "- Missing validations or edge cases",
    "- Code quality issues that could cause future bugs",
    "",
    "For each issue: provide the exact filename, the exact line number from the prefix (N), the line_type",
    "(ADDED/REMOVED/CONTEXT), severity (CRITICAL/HIGH/MEDIUM/LOW), a clear description, and a concrete suggestion.",
    "",
    "Only report real issues — avoid trivial style comments.",
    "Respond with JSON matching the schema provided."
  ].join("\n");
}

/**
 * Builds the aggregate summary prompt for generating the final PR verdict.
 * @param {object} prMeta - PR metadata
 * @param {Array} fileReviews - Array of file review results with filepath and review
 */
export function getAggregateSummaryPrompt(prMeta, fileReviews) {
  const issuesSections = (fileReviews || []).map(fr => {
    const issues = fr.review?.potential_issues || [];
    if (issues.length === 0) return `### ${fr.filepath}\nNo issues found.`;
    return [
      `### ${fr.filepath}`,
      ...issues.map((issue, i) => {
        const locs = (issue.locations || [])
          .map(l => `${l.filename}:${l.line_number}`)
          .join(", ");
        return `${i + 1}. [${issue.severity}] ${issue.issue_description}\n   Location: ${locs || "unknown"}`;
      })
    ].join("\n");
  }).join("\n\n");

  return [
    "You are a senior software engineer summarizing a complete pull request review.",
    "",
    "## Pull Request",
    `- Project: ${prMeta.project}`,
    `- Repository: ${prMeta.repo}`,
    `- PR ID: ${prMeta.pr}`,
    "",
    "## Review Findings",
    issuesSections || "No issues found across all reviewed files.",
    "",
    "## Task",
    "Based on the review findings above:",
    "1. Write a concise overall summary of what the PR changes and its overall quality.",
    "2. List the most important issues (deduplicated, sorted by severity, max 20).",
    "3. Provide a verdict: 'Approve' if the PR is acceptable, or 'Request-changes' with a brief reason.",
    "",
    "Respond with JSON matching the schema provided."
  ].join("\n");
}

/**
 * Builds the issue normalization prompt for deduplicating and filtering LLM review findings.
 * @param {object} prMeta - PR metadata
 * @param {Array} issues - Array of issue objects to normalize
 */
export function getIssuesNormalizationPrompt(prMeta, issues) {
  const issuesList = Array.isArray(issues) && issues.length > 0
    ? issues.map((issue, i) => {
        const locs = (issue.locations || [])
          .map(l => `${l.filename}:${l.line_number}`)
          .join(", ");
        const suggestion = issue.suggestion?.text || "(none)";
        return `${i + 1}. [${issue.severity}] ${issue.issue_description}\n   Locations: ${locs || "unknown"}\n   Suggestion: ${suggestion}`;
      }).join("\n\n")
    : "No issues to normalize.";

  return [
    "You are a senior software engineer normalizing code review findings.",
    "",
    "## Pull Request",
    `- Project: ${prMeta.project}`,
    `- Repository: ${prMeta.repo}`,
    `- PR ID: ${prMeta.pr}`,
    "",
    "## Issues to Normalize",
    issuesList,
    "",
    "## Task",
    "Review the list of issues above and:",
    "1. Remove exact duplicates — keep only one instance of each unique issue.",
    "2. Merge issues that describe the same underlying problem at overlapping locations.",
    "3. Recalibrate severity if it seems too high or too low based on actual impact.",
    "4. Return only the top 20 most impactful and actionable issues.",
    "5. Preserve the original issue_description, suggestion, and locations for each kept issue.",
    "",
    "Respond with JSON matching the schema provided."
  ].join("\n");
}

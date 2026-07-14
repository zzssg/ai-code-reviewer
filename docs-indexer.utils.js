// -----------------------------
// CONFLUENCE STORAGE FORMAT -> TEXT
// -----------------------------
const ENTITY_MAP = {
  "&nbsp;": " ",
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&apos;": "'",
  "&#39;": "'",
  "&ndash;": "–",
  "&mdash;": "—",
  "&hellip;": "…"
};

function decodeEntities(text) {
  return text
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, code) => String.fromCharCode(parseInt(code, 16)))
    .replace(/&[a-zA-Z]+;|&#\d+;/g, entity => ENTITY_MAP[entity] ?? " ");
}

/**
 * Converts Confluence storage-format XHTML (body.storage.value) into plain text.
 * Code macro bodies (CDATA) are preserved as-is, block-level tags become line
 * breaks, all other markup is stripped.
 */
export function storageToText(html) {
  if (!html) return "";

  let text = html;

  // Preserve code / plain-text macro bodies
  text = text.replace(
    /<ac:plain-text-body>\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*<\/ac:plain-text-body>/gi,
    "\n$1\n"
  );
  // Unwrap any remaining CDATA sections
  text = text.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1");

  // Block-level elements produce line breaks, list items become bullets
  text = text.replace(/<br\s*\/?>/gi, "\n");
  text = text.replace(/<li[^>]*>/gi, "\n- ");
  text = text.replace(/<\/(p|div|li|tr|table|h[1-6]|blockquote|pre|ul|ol)>/gi, "\n");
  text = text.replace(/<\/t[dh]>/gi, " | ");

  // Strip all remaining tags, including namespaced ac:/ri: macro tags
  text = text.replace(/<[^>]+>/g, "");

  text = decodeEntities(text);

  // Collapse whitespace noise
  text = text
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return text;
}

// -----------------------------
// PAGE CHUNKING BY HEADINGS
// -----------------------------
/**
 * Splits a Confluence page (storage-format XHTML) into chunks by h1-h6 headings.
 * Content before the first heading becomes an intro chunk titled after the page.
 * Falls back to a single "full_page" chunk when the page has no headings.
 *
 * @returns {Array<{chunk_id: number, heading: string, level: number, text: string}>}
 */
export function chunkPageByHeadings(storageHtml, pageTitle = "") {
  const html = storageHtml || "";
  const headingRegex = /<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi;

  const boundaries = [];
  let match;
  while ((match = headingRegex.exec(html)) !== null) {
    boundaries.push({
      level: Number(match[1]),
      heading: storageToText(match[2]) || "untitled_section",
      start: match.index,
      bodyStart: match.index + match[0].length
    });
  }

  const chunks = [];
  let chunkId = 0;

  const pushChunk = (heading, level, sectionHtml) => {
    const body = storageToText(sectionHtml);
    const text = [heading, body].filter(Boolean).join("\n");
    if (!text.trim()) return;
    chunks.push({ chunk_id: chunkId++, heading, level, text });
  };

  if (boundaries.length === 0) {
    const text = storageToText(html);
    if (text) {
      chunks.push({ chunk_id: 0, heading: pageTitle || "full_page", level: 0, text });
    }
    return chunks;
  }

  // Intro content before the first heading
  if (boundaries[0].start > 0) {
    pushChunk(pageTitle || "introduction", 0, html.slice(0, boundaries[0].start));
  }

  for (let i = 0; i < boundaries.length; i++) {
    const sectionEnd = i + 1 < boundaries.length ? boundaries[i + 1].start : html.length;
    pushChunk(
      boundaries[i].heading,
      boundaries[i].level,
      html.slice(boundaries[i].bodyStart, sectionEnd)
    );
  }

  return chunks;
}

import axios from "axios";
import {
  createLogger,
  embedText,
  getOsClient,
  fileChecksum,
  runLimited,
  percentile,
  ensureIndex
} from "./utils.js";
import { chunkPageByHeadings, storageToText } from "./docs-indexer.utils.js";

const log = createLogger(import.meta.url);

// Flag to skip cleanup of obsolete pages and chunks
const SKIP_CLEANUP = process.env.SKIP_CLEANUP === "true";
// Opensearch index name for Confluence docs
const DOCS_INDEXER_INDEX_NAME = process.env.DOCS_INDEXER_INDEX_NAME || "confluence-docs-embeddings-chunks";

const CONFLUENCE_TOKEN = process.env.CONFLUENCE_TOKEN;
const CONFLUENCE_BASE = process.env.CONFLUENCE_BASE || "https://confluence/rest/api";
const CONFLUENCE_HEADERS = { Authorization: `Bearer ${CONFLUENCE_TOKEN}` };
// Comma-separated list of Confluence space keys to index, e.g. "DEV,ARCH"
const CONFLUENCE_SPACES = (process.env.CONFLUENCE_SPACES || "")
  .split(",")
  .map(s => s.trim())
  .filter(Boolean);
const PAGE_FETCH_LIMIT = 50;

// -----------------------------
// BLACKLIST PAGE TITLES
// -----------------------------
const blackList = [
  "Meeting notes",
  "Page templates",

];

// -----------------------------
// PAGE COLLECTION
// -----------------------------
async function getSpacePages(spaceKey) {
  const pages = [];
  let start = 0;

  while (true) {
    const url = `${CONFLUENCE_BASE}/content` +
      `?spaceKey=${encodeURIComponent(spaceKey)}` +
      `&type=page&status=current` +
      `&expand=body.storage,version,ancestors,metadata.labels` +
      `&start=${start}&limit=${PAGE_FETCH_LIMIT}`;

    const res = await axios.get(url, { headers: CONFLUENCE_HEADERS });
    const data = res.data;
    const results = data.results || [];
    const baseLink = data._links?.base || "";

    for (const page of results) {
      if (blackList.includes(page.title)) continue;
      pages.push({
        pageId: page.id,
        space: spaceKey,
        title: page.title,
        storageHtml: page.body?.storage?.value || "",
        version: page.version?.number || 0,
        labels: (page.metadata?.labels?.results || []).map(l => l.name),
        ancestors: (page.ancestors || []).map(a => a.title),
        url: page._links?.webui ? baseLink + page._links.webui : ""
      });
    }

    if (results.length < PAGE_FETCH_LIMIT || !data._links?.next) break;
    start += PAGE_FETCH_LIMIT;
  }

  return pages;
}

// -----------------------------
// PAGE IMPORTANCE
// -----------------------------
function computeImportance(page, textContent) {
  let score = 1.0;
  const staleMarkers = ["archived", "deprecated", "obsolete", "draft"];
  const title = page.title.toLowerCase();
  if (page.ancestors.length <= 1) score += 0.5; // top-level pages are usually entry points
  if (page.labels.some(l => staleMarkers.includes(l.toLowerCase()))) score -= 0.5;
  if (staleMarkers.some(m => title.includes(m))) score -= 0.5;
  const lines = textContent.split(/\r?\n/).length;
  if (lines < 200) score += 0.5;
  return Math.max(score, 0.1);
}

// -----------------------------
// GET STORED CHECKSUMS
// -----------------------------
async function getStoredPageChecksum(docId) {
  try {
    const res = await getOsClient().get({ index: DOCS_INDEXER_INDEX_NAME, id: docId });
    return res.body?._source?.checksum ?? null;
  } catch {
    return null;
  }
}

// -----------------------------
// OBSOLETE PAGE + CHUNK CLEANUP
// -----------------------------
async function cleanupObsoletePages(existingPageIds) {
  const os = getOsClient();
  const existingSet = new Set(existingPageIds);

  log.info("Starting obsolete page cleanup...");
  const startTS = Date.now();

  let deletedPages = 0;
  let deletedChunks = 0;
  let searchAfter = null;

  while (true) {
    const res = await os.search({
      index: DOCS_INDEXER_INDEX_NAME,
      size: 500,
      body: {
        sort: [{ page_id: "asc" }],
        ...(searchAfter ? { search_after: searchAfter } : {}),
        query: {
          term: { doc_type: "page" }
        }
      }
    });

    const hits = res.body.hits.hits;
    if (hits.length === 0) break;

    for (const hit of hits) {
      const pageId = hit._source.page_id;

      if (!existingSet.has(pageId)) {
        // delete parent page doc
        await os.delete({
          index: DOCS_INDEXER_INDEX_NAME,
          id: hit._id
        }).catch(() => {});
        deletedPages++;

        // delete all chunks for this page
        const delRes = await os.deleteByQuery({
          index: DOCS_INDEXER_INDEX_NAME,
          refresh: true,
          body: {
            query: {
              bool: {
                must: [
                  { term: { doc_type: "chunk" } },
                  { term: { page_id: pageId } }
                ]
              }
            }
          }
        });

        deletedChunks += delRes.body.deleted || 0;
        log.info(`Deleted obsolete page + chunks: ${pageId} (${hit._source.title})`);
      }
    }

    searchAfter = hits[hits.length - 1].sort;
  }

  log.info(
    `Cleanup finished in ${Date.now() - startTS} ms. Removed pages / chunks: ${deletedPages} / ${deletedChunks}`
  );
}

// -----------------------------
// MAIN INDEXING
// -----------------------------
async function indexSpaces(spaceKeys) {
  let totalChunks = 0;
  let embedCalls = 0;
  let embedLatencies = [];
  let skippedDuplicates = 0;

  log.info("Ensuring OpenSearch index exists...");
  await ensureIndex(DOCS_INDEXER_INDEX_NAME, "./data/create-opensearch-index-docs.json");
  log.info(`Starting indexing of Confluence spaces [${spaceKeys.join(", ")}], using Opensearch index ${DOCS_INDEXER_INDEX_NAME}...`);

  const pages = [];
  for (const spaceKey of spaceKeys) {
    const spacePages = await getSpacePages(spaceKey);
    log.info(`Space ${spaceKey}: found ${spacePages.length} pages to process.`);
    pages.push(...spacePages);
  }
  log.info(`Found ${pages.length} pages total.`);

  const existingPageIds = pages.map(p => p.pageId);

  const tasks = pages.map(page => async () => {
    const textContent = storageToText(page.storageHtml);
    const pageHash = fileChecksum(page.storageHtml);

    // Skip unchanged pages entirely
    const existingPageChecksum = await getStoredPageChecksum(page.pageId);
    if (existingPageChecksum === pageHash) {
      skippedDuplicates++;
      return; // do not process chunks or embeddings
    }

    let taskLatency = { chunks: 0, latency: 0 };

    // Chunk detection
    const chunks = chunkPageByHeadings(page.storageHtml, page.title);

    const headings = chunks.map(c => c.heading);
    const importance = computeImportance(page, textContent);

    // Store parent page doc
    await getOsClient().index({
      index: DOCS_INDEXER_INDEX_NAME,
      id: page.pageId,
      body: {
        doc_type: "page",
        space: page.space,
        page_id: page.pageId,
        title: page.title,
        labels: page.labels,
        ancestors: page.ancestors,
        url: page.url,
        version: page.version,
        content: textContent,
        headings,
        checksum: pageHash,
        importance,
        chunks_count: chunks.length
      }
    });

    // Process chunks (we know page is new/changed, so all chunks need embedding)
    for (const chunk of chunks) {
      totalChunks++;
      const chunkId = `${page.pageId}::chunk_${chunk.chunk_id}`;
      const chunkHash = fileChecksum(chunk.text);

      const t0 = Date.now();
      const embeddings = await embedText(chunk.text);
      embedCalls++;
      taskLatency.chunks++;
      taskLatency.latency += (Date.now() - t0);

      if (embeddings.length <= 1) {
        await getOsClient().index({
          index: DOCS_INDEXER_INDEX_NAME,
          id: chunkId,
          body: {
            doc_type: "chunk",
            space: page.space,
            page_id: page.pageId,
            title: page.title,
            labels: page.labels,
            ancestors: page.ancestors,
            url: page.url,
            version: page.version,
            chunk_id: chunk.chunk_id,
            heading: chunk.heading,
            level: chunk.level,
            content: chunk.text,
            checksum: chunkHash,
            embedding: embeddings[0] || [],
            importance
          }
        });
      } else {
        // In case embedding model returns multiple vectors (e.g. for long chunks), we store them as separate docs with same chunk_id but different sub_chunk_id
        for (let i = 0; i < embeddings.length; i++) {
          const subChunkId = `${chunkId}_sub_${i}`;
          await getOsClient().index({
            index: DOCS_INDEXER_INDEX_NAME,
            id: subChunkId,
            body: {
              doc_type: "chunk",
              space: page.space,
              page_id: page.pageId,
              title: page.title,
              labels: page.labels,
              ancestors: page.ancestors,
              url: page.url,
              version: page.version,
              chunk_id: chunk.chunk_id,
              sub_chunk_id: i,
              heading: chunk.heading,
              level: chunk.level,
              content: chunk.text,
              checksum: chunkHash,
              embedding: embeddings[i],
              importance
            }
          });
        }
      }
    }
    embedLatencies.push(taskLatency);
    log.info(`Processed page: ${page.space}/${page.title} (${page.pageId}), chunks: ${taskLatency.chunks}, avg latency: ${taskLatency.latency / taskLatency.chunks || 0} ms, total latency: ${taskLatency.latency} ms`);
  });

  log.info(`Prepared ${tasks.length} tasks. Starting indexing...`);
  const started = Date.now();
  await runLimited(tasks, 1);
  const duration = Date.now() - started;

  embedLatencies.sort((a, b) => a.latency - b.latency);
  const min = embedLatencies[0]?.latency ?? 0;
  const max = embedLatencies[embedLatencies.length - 1]?.latency ?? 0;
  const p50 = percentile(embedLatencies.map(l => l.latency), 50);

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

  if (SKIP_CLEANUP) {
    log.info(`Skipping obsolete page cleanup as per configuration: SKIP_CLEANUP is ${SKIP_CLEANUP}`);
    return;
  } else {
    await cleanupObsoletePages(existingPageIds);
  }
}

// run
if (!CONFLUENCE_TOKEN) {
  log.error("CONFLUENCE_TOKEN is not set");
  process.exit(1);
}
if (CONFLUENCE_SPACES.length === 0) {
  log.error("CONFLUENCE_SPACES is not set, expected comma-separated space keys, e.g. CONFLUENCE_SPACES=DEV,ARCH");
  process.exit(1);
}
await indexSpaces(CONFLUENCE_SPACES);

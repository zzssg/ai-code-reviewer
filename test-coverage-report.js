/**
 * Standalone script to compute semantic test coverage from OpenSearch index embeddings
 */

import {
  getOsClient,
  INDEX_NAME,
  EMB_SIZE
} from "./utils.js";

import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

const client = getOsClient();

// cosine similarity between two normalized vectors
function cosineSimilarity(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// normalize a vector
function normalize(vec) {
  const len = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0));
  return len > 0 ? vec.map(v => v / len) : vec;
}

// compute centroid of array of vectors
function computeCentroid(vectors) {
  const centroid = new Array(EMB_SIZE).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < EMB_SIZE; i++) centroid[i] += vec[i];
  }
  for (let i = 0; i < EMB_SIZE; i++) centroid[i] /= vectors.length;
  return normalize(centroid);
}

// fetch all docs from OpenSearch index
async function fetchAllDocs() {
  log.info(`Loading all documents from index '${INDEX_NAME}'...`);
  const t0 = Date.now();
  const allDocs = [];
  let resp = await client.search({
    index: INDEX_NAME,
    scroll: "2m",
    size: 1000,
    body: { query: { match_all: {} } }
  });

  allDocs.push(...resp.body.hits.hits.map(h => ({ id: h._id, ...h._source })));
  let scrollId = resp.body._scroll_id;

  while (true) {
    resp = await client.scroll({ scroll_id: scrollId, scroll: "2m" });
    if (!resp.body.hits.hits.length) break;
    allDocs.push(...resp.body.hits.hits.map(h => ({ id: h._id, ...h._source })));
    scrollId = resp.body._scroll_id;
  }

  log.info(`Loaded ${allDocs.length} documents in ${Date.now() - t0} ms`);
  return allDocs;
}

async function generateCoverageReport() {
  log.info(`Start generating coverage report for index '${INDEX_NAME}'...`);
  const allDocs = await fetchAllDocs();

  const sourceDocs = allDocs.filter(d => !d.is_test);
  const testDocsRaw = allDocs.filter(d => d.is_test);

  log.info(`Source chunks: ${sourceDocs.length}`);
  log.info(`Test chunks:   ${testDocsRaw.length}`);

  const testDocs = testDocsRaw.filter(d => Array.isArray(d.embedding) && d.embedding.length === EMB_SIZE);
  if (!testDocs.length) {
    log.info("No valid test embeddings found. Exiting.");
    return;
  }

  log.info("Computing semantic coverage...");

  // centroid for all test embeddings
  const testVecs = testDocs.map(d => normalize(d.embedding));
  const testCentroid = computeCentroid(testVecs);

  // compute coverage per source chunk and update in OpenSearch
  const coverageMap = [];

  await Promise.all(
    sourceDocs
      .filter(d => Array.isArray(d.embedding) && d.embedding.length === EMB_SIZE)
      .map(async (src) => {
        const sim = cosineSimilarity(normalize(src.embedding), testCentroid);
        const coveragePercent = Math.round(sim * 100);
        coverageMap.push({ id: src.id, similarity: sim, coveragePercent });

        // update document in OpenSearch with coveragePercent
        await client.update({
          index: INDEX_NAME,
          id: src.id,
          body: {
            doc: { coverage_percent: coveragePercent }
          }
        });
      })
  );

  // sort by similarity descending
  coverageMap.sort((a, b) => b.similarity - a.similarity);

  const avgCoverage = coverageMap.reduce((acc, c) => acc + c.similarity, 0) / coverageMap.length;
  log.info(`Average test coverage over ${INDEX_NAME}: ${(avgCoverage * 100).toFixed(2)}%`);

  // top 10 most covered chunks
  const top10 = coverageMap.slice(0, 10);
  log.info("Top 10 covered source chunks:");
  top10.forEach((c, i) => {
    log.info(`${i + 1}. ${c.id} -> ${c.coveragePercent}%`);
  });
}

generateCoverageReport().catch(err => {
  log.error("Error generating test coverage report for index: " + INDEX_NAME, err);
});

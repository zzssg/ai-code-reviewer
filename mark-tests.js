/**
 *  Standalone script to update all docs in an OpenSearch index
 *  Adds boolean field `is_test` based on filename/filepath
 * */ 

import {
  getOsClient,
  INDEX_NAME
} from "./utils.js";

import createLogger from "./logger.js";
const log = createLogger(import.meta.url);

function isTestFile(doc) {
  const filename = doc.filename || "";
  const filepath = doc.filepath || "";

  const lowerPath = filepath.toLowerCase();

  if (lowerPath.includes("/test/") || lowerPath.includes("\\test\\")) {
    return true;
  }

  if (filename.endsWith("Test.java") || filename.endsWith("Tests.java")) {
    return true;
  }

  return false;
}

async function run() {
  const client = getOsClient();

  log.info(`Scanning index: ${INDEX_NAME}`);
  const startTS = Date.now();
  const first = await client.search({
    index: INDEX_NAME,
    scroll: "2m",
    size: 500,
    body: {
      query: { match_all: {} }
    }
  });

  let scrollId = first.body._scroll_id;
  let hits = first.body.hits.hits;

  let processed = 0;
  let isTestCount = 0;

  while (hits.length > 0) {
    for (const hit of hits) {
      const id = hit._id;
      const source = hit._source;

      const newIsTest = isTestFile(source);
      if (newIsTest) {
        isTestCount++;
      }

      // Update document with "is_test" flag
      await client.update({
        index: INDEX_NAME,
        id,
        body: {
          doc: { is_test: newIsTest }
        }
      });

      processed++;
      if (processed % 500 === 0) {
        log.info(`Processed: ${processed} docs...`);
      }
    }

    const next = await client.scroll({
      scroll_id: scrollId,
      scroll: "2m"
    });

    scrollId = next.body._scroll_id;
    hits = next.body.hits.hits;
  }

  log.info(`Done. Documents updated: ${processed}. Tests found: ${isTestCount}. Time spent: ${(Date.now() - startTS)/1000} sec`);
}

run().catch(err => {
  log.error(`Error while marking docs in "${INDEX_NAME}" index with "is_test" flag:`, err);
});

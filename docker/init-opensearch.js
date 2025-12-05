import { Client } from "@opensearch-project/opensearch";

const osClient = new Client({ node: "http://localhost:9200" });

const INDEX_NAME = "repo-code-embeddings";
const EMBEDDING_DIM = 768;

async function initOpenSearch() {
  const exists = await osClient.indices.exists({ index: INDEX_NAME });
  if (!exists.body) {
    await osClient.indices.create({
      index: INDEX_NAME,
      body: {
        settings: { index: { knn: true } },
        mappings: {
          properties: {
            filename: { type: "keyword" },
            filepath: { type: "keyword" },
            language: { type: "keyword" },
            content: { type: "text" },
            embedding: {
              type: "knn_vector",
              dimension: EMBEDDING_DIM,
              method: { name: "hnsw", engine: "nmslib", space_type: "cosinesimil" }
            }
          }
        }
      }
    });
  }
}

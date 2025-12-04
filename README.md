# Local Embedding Server

A minimal Express-based API for generating text embeddings using a **locally stored** Transformer model (`all-MiniLM-L6-v2`).

## Setup

```bash
npm install
```
## Run single embedding server

```bash
node embedding-server.js
```

## Run embedding server cluster

```bash
# To run cluster with 4 workers
EMB_WORKERS=4 node embedding-server-cluster-chunks.js

# If WORKERS param is ommited - single worker would be serving the cluster
```
## API

**POST** `/api/embedding`

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

### Integrations
***Embeddings storage***
Opensearch could be used to store embeddings produced by this embeddings-server.
To create Opensearch index for such 384D embeddings you can use following Opensearch API request:
```bash
curl -X PUT "http://localhost:9200/repo-code-embeddings" -u <user>:<pass> -H "Content-Type: application/json" -d @create-opensearch-index.json
```
Where the payload denoted by create-opensearch-index.json is:
```json
{
    "settings": {
      "index": {
        "knn": true
      }
    },
    "mappings": {
      "properties": {
        "filename": { "type": "keyword" },
        "filepath": { "type": "keyword" },
        "language": { "type": "keyword" },
        "checksum": { "type": "text" },
        "content": { "type": "text" },
        "embedding": {
          "type": "knn_vector",
          "dimension": 384,
          "method": {
            "name": "hnsw",
            "engine": "faiss",
            "space_type": "cosinesimil"
          }
        }
      }
    }
}
```

# Notes
This project is based on all-MiniLM-L6-v2 (Xenova / Hugging Face) under Apache 2.0 licence.

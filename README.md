# AI Code Reviewer

An intelligent code review system that leverages local embeddings and Retrieval-Augmented Generation (RAG) to provide contextual code reviews. This system consists of three main components:

## Core Features

### 1. Embedding Server
**File:** `embedding-server-cluster-chunks.js`

A high-performance Express-based API for generating text embeddings using a locally stored Transformer model (`all-MiniLM-L6-v2`). The server processes source code and generates 384-dimensional embeddings that capture semantic meaning.

Key features:
- Cluster-based architecture for improved performance
- Text chunking for handling large code segments
- ONNX runtime for efficient model inference
- RESTful API endpoint at `/api/embedding`

### 2. Repository Indexer
**File:** `repo-indexer-chunks.js`

Scans source code repositories, creates embeddings for code chunks, and stores them in OpenSearch for later retrieval. 

Process:
1. Recursively scans directories for source code files
2. Intelligently chunks code by functions/methods based on language
3. Generates embeddings for each chunk using the embedding server
4. Stores embeddings and metadata in OpenSearch with efficient indexing

Supports multiple languages including JavaScript, TypeScript, Java, and Python with language-specific parsing.

### 3. Pull Request Reviewer
**File:** `pr-reviewer-chunks.js`

Implements Retrieval-Augmented Generation (RAG) for automated pull request reviews. Listens for BitBucket webhook events and performs contextual code reviews.

Process:
1. Receives pull request events via BitBucket webhook
2. Extracts code changes and pull request description
3. Finds semantically similar code chunks from the indexed repository
4. Constructs context-aware prompts for the LLM
5. Posts AI-generated review comments back to the pull request

This approach significantly improves review quality by providing the LLM with relevant context from the existing codebase.

## Auxiliary Features

### Test Code Identification
**File:** `mark-tests.js`

Automatically identifies and marks test code in the OpenSearch index with an `is_test` flag. Uses filepath patterns and naming conventions to distinguish test files from production code.

### Semantic Test Coverage Analysis
**File:** `test-coverage-report.js`

Analyzes the semantic relationship between test code and production code to calculate coverage percentages. Computes a "coverage_percent" value for each production code chunk based on its semantic similarity to existing test code.

## Setup

```bash
npm install
```

### Running the Embedding Server

```bash
# To run cluster with specified number of workers
EMB_WORKERS=4 node embedding-server-cluster-chunks.js

# If EMB_WORKERS param is omitted, defaults to CPU-based worker count
node embedding-server-cluster-chunks.js
```

### Indexing a Repository

```bash
# Set the path to the repository you want to index
EMB_PATH_TO_REPO=/path/to/your/repository node repo-indexer-chunks.js
```

### Running the PR Reviewer

```bash
# Start the PR reviewer service
node pr-reviewer-chunks.js
```

### Marking Test Files

```bash
# Mark files as test code in the index
node mark-tests.js
```

### Generating Test Coverage Report

```bash
# Generate semantic test coverage report
node test-coverage-report.js
```

## API

**POST** `/api/embedding`

Generates embeddings for the provided text.

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

## Integrations

### Embeddings Storage
OpenSearch is used to store the generated embeddings. To create an OpenSearch index for 384-dimensional embeddings:

```bash
curl -X PUT "http://localhost:9200/repo-code-embeddings" -u <user>:<pass> -H "Content-Type: application/json" -d @create-opensearch-index-chunks.json
```

The index configuration supports KNN (k-nearest neighbors) search for efficient similarity lookups.

## Notes
This project is based on all-MiniLM-L6-v2 (Xenova / Hugging Face) under Apache 2.0 license.
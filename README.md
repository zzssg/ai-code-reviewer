# Local Embedding Server

A minimal Express-based API for generating text embeddings using a **locally stored** Transformer model (`all-MiniLM-L6-v2`) via `@xenova/transformers`.

## Setup

```bash
npm install express @xenova/transformers
```
## Run

```bash
node embedding-server.js
```
## API

**POST** `/api/embedding`

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

# Notes
This project is based on all-MiniLM-L6-v2 (Xenova / Hugging Face) under Apache 2.0 licence.

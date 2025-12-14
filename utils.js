import axios from "axios";
import crypto from "crypto";
import { Client } from "@opensearch-project/opensearch";

export const INDEX_NAME = process.env.EMB_INDEX_NAME || "repo-code-embeddings-chunks";
export const PATH_TO_REPO = process.env.EMB_PATH_TO_REPO || "../OpenSearch/";
export const EMB_ENDPOINT = process.env.EMB_ENDPOINT || "http://localhost:3000/api/embedding";
export const OS_ENDPOINT = process.env.EMB_OS_ENDPOINT ||"http://localhost:9200";
export const LLM_REVIEW_MODEL = process.env.LLM_REVIEW_MODEL || "qwen3-coder-30b-a3b-instruct-ud"; //"devstral-small-2-24b-instruct-2512"; // "qwen3-coder-30b-a3b-instruct-ud";
export const LLM_ENDPOINT = process.env.LLM_ENDPOINT || "http://localhost:1234/v1/responses";
export const LLM_API_KEY = process.env.LLM_API_KEY || "";

export const EMB_SIZE = 384; // Embedding size

let osClient = undefined;

export async function embedText(text) {
  const response = await fetch(EMB_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding request failed: ${response.status} ${errText}`);
  }

  const data = await response.json();

  if (!data.embeddings) {
    throw new Error("Invalid response: missing 'embeddings' field");
  }

  return data.embeddings;
}

export function prepareOpensearchIndexName() {
  const normalized = path.normalize(PATH_TO_REPO).replace(/[\\/]+$/, "");
  const index_suffix = path.basename(normalized).toLowerCase();
  let result = INDEX_NAME;
  if (index_suffix && index_suffix.length > 0) {
    result = INDEX_NAME + "-" + index_suffix;
  }
  return result;
}

export async function searchContext(queryText) {
  const vector = await embedText(queryText);
  const osClient = getOsClient();
  const res = await osClient.search({
    index: INDEX_NAME,
    body: {
      size: 10,
      query: {
        knn: {
          embedding: {
            vector,
            k: 10
          }
        }
      }
    }
  });
  return res.body.hits.hits.map(hit => hit._source);
}

export async function queryLLM(prompt) {
  try {
    const payload = {
      "model": LLM_REVIEW_MODEL,
      "input": prompt,
      "stream": false
    };
    const headers = { "Content-Type": "application/json" };
    if (LLM_API_KEY != "") {
      headers["Authorization"] = `Bearer ${LLM_API_KEY}`;
    }
    const LlmRawResult = await axios.post(LLM_ENDPOINT, payload, { headers });
    return LlmRawResult.data.output[0]?.content[0]?.text || "NO RESPONSE FROM LLM";
  } catch (err) {
    console.error("LLM query failed: ", err);
    return "LLM QUERY FAILED";
  }
    
}

export async function reviewPullRequest({ description, diff, context }) {
  const prompt = `
You are a senior code reviewer.

Project context (summarized):
${context.map(c => `File: ${c.filepath}\n${c.content.slice(0, 300)}`).join("\n\n")}

Pull request description:
${description}

Code diff:
${diff}

Please provide a detailed review with:
- potential bugs or logic errors
- code quality or readability issues
- suggestions for improvement
  `;

  const payload = {
      "model": "gpt-oss:120b",
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ]
    };
  return payload;
}

export function getOsClient() {
  if (!osClient) osClient = new Client({ node: OS_ENDPOINT });
  return osClient;
}

export function fileChecksum(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

export async function runLimited(tasks, limit = 4) {
  const results = [];
  let i = 0;

  async function worker() {
    while (i < tasks.length) {
      const taskIndex = i++;
      results[taskIndex] = await tasks[taskIndex]();
    }
  }

  const workers = Array.from({ length: limit }, worker);
  await Promise.all(workers);

  return results;
}

export function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const index = Math.floor((p / 100) * arr.length);
  return arr[index];
}
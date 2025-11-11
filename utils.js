export const INDEX_NAME = "repo-code-embeddings";
export const PATH_TO_REPO = "../OpenSearch/";

let osClient = undefined;

export async function embedText(text) {
  const response = await fetch("http://localhost:3000/api/embedding", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Embedding request failed: ${response.status} ${errText}`);
  }

  const data = await response.json();

  if (!data.embedding) {
    throw new Error("Invalid response: missing 'embedding' field");
  }

  return data.embedding;
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

export function queryLLM(prompt) {
  const payload = [{ role: "user", content: prompt }];
  // Should be implemented reall call to LLM
  return "This is test response emulating PR review results provided by LLM";
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

  const res = queryLLM(prompt);
  return res;
}

export function getOsClient() {
  if (!osClient) osClient = new Client({ node: "http://localhost:9200" });
  return osClient;
}
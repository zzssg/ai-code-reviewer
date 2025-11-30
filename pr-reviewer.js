import express from "express";
import axios from "axios";
import { searchContext, reviewPullRequest } from "./utils.js";

// ---------------------------------------------------------------------------
// LOGGER SETUP
// ---------------------------------------------------------------------------
import createLogger from "./logger.js";
const log = createLogger(import.meta.url);
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json());

app.post("/bitbucket/pr-event", async (req, res) => {
  const event = req.body;
  const pr = event.pullrequest;
  const prId = pr.id;
  const description = pr.description || "";
  const diffUrl = pr.links.diff.href;

  // Fetch diff text
  const diffRes = await axios.get(diffUrl);
  const diff = diffRes.data;

  // Retrieve related context
  const context = await searchContext(description + "\n" + diff);

  // Ask the LLM for review
  const review = await reviewPullRequest({ description, diff, context });

  // Post back to Bitbucket
  await axios.post(
    `https://api.bitbucket.org/2.0/repositories/${pr.destination.repository.full_name}/pullrequests/${prId}/comments`,
    { content: { raw: review } },
    { headers: { Authorization: `Bearer ${process.env.BITBUCKET_TOKEN}` } }
  );

  res.sendStatus(200);
});

app.listen(3000, () => log.info("PR Review Bot listening on :3000"));

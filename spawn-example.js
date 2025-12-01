import { spawn } from "node:child_process";

spawn(
  "node",
  ["/../../repo-indexer.js"],
  {
    cwd: "/../../embedding-server",
    stdio: "inherit",
    env: {
      ...process.env,                               // keep existing env
      PATH_TO_REPO: "/some/path/from/teamcity",     // add your own env var
      NODE_ENV: "production"                        // more vars if needed
    }
  }
);

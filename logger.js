import pino from "pino";
import { fileURLToPath } from "url";
import path from "path";

const isDev = process.env.NODE_ENV !== "production";

export default function createLogger(importMetaUrl) {
  const filename = path.basename(fileURLToPath(importMetaUrl));

  const logger = pino({
    level: "info",
    timestamp: pino.stdTimeFunctions.isoTime,
    formatters: {
      level(label) {
        return { level: label.toUpperCase() };
      },
      bindings() {
        return {};
      },
      log(obj) {
        return obj;
      }
    },
  }, isDev
    ? pino.transport({
        target: "pino-pretty",
        options: {
          colorize: false,
          translateTime: false,
          messageFormat: `${filename} - {msg}`,
        },
      })
    : undefined
  );

  return logger;
}
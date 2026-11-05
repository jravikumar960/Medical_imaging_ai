import test from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));

test("status messages use semantic color classes and accessible roles", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  try {
    const { default: StatusMessage } = await vite.ssrLoadModule("/src/StatusMessage.jsx");
    const render = (kind) => renderToStaticMarkup(
      createElement(StatusMessage, { kind }, "Status"),
    );

    assert.match(render("success"), /class="success-message" role="status"/);
    assert.match(render("error"), /class="error" role="alert"/);
    assert.match(render("warning"), /class="warning-message" role="status"/);
    assert.match(render("info"), /class="info-message" role="status"/);
    assert.throws(
      () => renderToStaticMarkup(createElement(StatusMessage, { kind: "pending" }, "Status")),
      /Unsupported status message kind: pending/,
    );
  } finally {
    await vite.close();
  }
});

import test from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));

test("React application renders the redesigned secure sign-in form when unauthenticated", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  try {
    const { default: App } = await vite.ssrLoadModule("/src/App.jsx");
    const html = renderToStaticMarkup(createElement(App));
    assert.match(html, /Secure workspace access/);
    assert.match(html, /Welcome back/);
    assert.match(html, /Username/);
    assert.match(html, /Password/);
    assert.match(html, /Sign in/);
    assert.match(html, /Register/);
    assert.match(html, /existing local review service/);
    assert.doesNotMatch(html, /Chest X-ray research analysis/);
  } finally {
    await vite.close();
  }
});

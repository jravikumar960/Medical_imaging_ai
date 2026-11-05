import test from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));

test("doctor review requires sign-in, shows status/assessment/notes fields", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  try {
    const { default: DoctorReviewPanel } = await vite.ssrLoadModule("/src/DoctorReviewPanel.jsx");
    const html = renderToStaticMarkup(createElement(DoctorReviewPanel, {
      studyId: "a".repeat(64),
      aiSummary: "A model mask was returned; this is not a diagnosis.",
      token: "",
    }));
    assert.match(html, /Doctor Review/);
    assert.match(html, /AI-generated study output/);
    assert.match(html, /A model mask was returned/);
    assert.match(html, /Pending/);
    assert.match(html, /Please sign in to access doctor review/);
  } finally {
    await vite.close();
  }
});

test("doctor review renders review fields when authenticated", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  try {
    const { default: DoctorReviewPanel } = await vite.ssrLoadModule("/src/DoctorReviewPanel.jsx");
    const html = renderToStaticMarkup(createElement(DoctorReviewPanel, {
      studyId: "a".repeat(64),
      aiSummary: "A model mask was returned; this is not a diagnosis.",
      token: "x".repeat(32),
    }));
    assert.match(html, /Doctor Review/);
    assert.match(html, /Review status/);
    assert.match(html, /Clinical assessment/);
    assert.match(html, /Review notes/);
    assert.match(html, /Pending/);
    assert.match(html, /In Review/);
    assert.match(html, /Reviewed/);
    assert.match(html, /Requires Further Assessment/);
    assert.match(html, /Save review/);
    assert.match(html, /Retrieve saved review/);
  } finally {
    await vite.close();
  }
});

import test from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));

test("modality selection clearly marks CT as awaiting independent review", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  try {
    const { default: DashboardOverview } = await vite.ssrLoadModule("/src/DashboardOverview.jsx");
    const html = renderToStaticMarkup(createElement(DashboardOverview, {
      username: "researcher",
      role: "researcher",
      services: [
        { id: "xray", label: "X-ray & review API", status: "offline", detail: "Offline" },
        { id: "ct", label: "CT segmentation API", status: "online", detail: "Connected" },
        { id: "mri", label: "MRI segmentation API", status: "online", detail: "Connected" },
      ],
      studies: [],
      studiesBusy: false,
      studiesError: "",
      onNavigate: () => {},
      onRefreshServices: () => {},
      onRefreshStudies: () => {},
    }));

    assert.match(html, /Chest X-ray/);
    assert.match(html, /MRI · Brain/);
    assert.match(html, /Pending independent review/);
    assert.match(html, /Research use only/);
    assert.match(html, /not be used for diagnosis or treatment/i);
  } finally {
    await vite.close();
  }
});

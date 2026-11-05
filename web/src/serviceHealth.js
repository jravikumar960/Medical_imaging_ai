const SERVICES = [
  {
    id: "xray",
    label: "X-ray & review API",
    baseUrl: import.meta.env?.VITE_XRAY_API_URL || "http://127.0.0.1:8000",
  },
  {
    id: "ct",
    label: "CT segmentation API",
    baseUrl: import.meta.env?.VITE_SEGMENTATION_API_URL || "http://127.0.0.1:8002",
  },
  {
    id: "mri",
    label: "MRI segmentation API",
    baseUrl: import.meta.env?.VITE_MRI_API_URL || "http://127.0.0.1:8003",
  },
];

export async function checkServiceHealth(fetchImpl = fetch) {
  return Promise.all(SERVICES.map(async ({ id, label, baseUrl }) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetchImpl(`${baseUrl.replace(/\/+$/, "")}/health`, {
        signal: controller.signal,
      });
      return {
        id,
        label,
        status: response.ok ? "online" : "offline",
        detail: response.ok ? "Health check passed" : `Health check returned HTTP ${response.status}`,
      };
    } catch (error) {
      return {
        id,
        label,
        status: "offline",
        detail: error?.name === "AbortError" ? "Health check timed out" : "Service could not be reached",
      };
    } finally {
      clearTimeout(timeout);
    }
  }));
}

const STATUS_CLASSES = {
  success: "success-message",
  error: "error",
  warning: "warning-message",
  info: "info-message",
};

export default function StatusMessage({ kind, children }) {
  const className = STATUS_CLASSES[kind];
  if (!className) {
    throw new Error(`Unsupported status message kind: ${kind}`);
  }

  return <p className={className} role={kind === "error" ? "alert" : "status"}>{children}</p>;
}

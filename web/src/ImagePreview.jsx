import { useEffect, useState } from "react";

export default function ImagePreview({ file, analyzed = false, previewPngBase64 = "" }) {
  const [url, setUrl] = useState("");
  const [zoom, setZoom] = useState(1);
  const [dimensions, setDimensions] = useState(null);

  useEffect(() => {
    if (!file) {
      setUrl("");
      setZoom(1);
      setDimensions(null);
      return undefined;
    }
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      setUrl(previewPngBase64 ? `data:image/png;base64,${previewPngBase64}` : "");
      setZoom(1);
      setDimensions(null);
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    setZoom(1);
    setDimensions(null);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, previewPngBase64]);

  if (!url) {
    return <p className="field-note">A pixel preview will appear after the server successfully processes this file.</p>;
  }

  return (
    <section className="image-preview" aria-label="Selected image preview">
      <div className="viewer-toolbar">
        <span>{analyzed ? "Analyzed source image" : "Selected image · not analyzed"}</span>
        <button type="button" className="icon-button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))}>−</button>
        <button type="button" className="icon-button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(3, value + 0.25))}>+</button>
        <button type="button" className="icon-button" aria-label="Reset zoom" onClick={() => setZoom(1)}>Reset</button>
      </div>
      <div className="image-preview-stage">
        <img src={url} alt={analyzed ? `Analyzed source image ${file.name}` : `Selected ${file.name}; analysis has not run`}
          onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
          style={{ transform: `scale(${zoom})` }} />
      </div>
      <dl className="ct-metadata">
        <div><dt>File</dt><dd>{file.name}</dd></div>
        <div><dt>Size</dt><dd>{(file.size / (1024 * 1024)).toFixed(2)} MiB</dd></div>
        <div><dt>Browser MIME type</dt><dd>{file.type || "unknown"}</dd></div>
        {dimensions && <div><dt>Pixel dimensions</dt><dd>{dimensions.width} × {dimensions.height}</dd></div>}
      </dl>
      {previewPngBase64 && !["image/png", "image/jpeg"].includes(file.type)
        && <p className="field-note">Server-generated pixel preview; DICOM metadata is not exposed.</p>}
    </section>
  );
}

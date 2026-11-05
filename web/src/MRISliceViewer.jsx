import { useEffect, useRef, useState } from "react";
import { composeMriSlice } from "./mriSlice.js";

export default function MRISliceViewer({ geometry, volumePreview, mask }) {
  const originalCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [width, height, depth] = geometry.shape;

  useEffect(() => {
    setSliceIndex(0);
    setZoom(1);
  }, [geometry]);

  useEffect(() => {
    for (const [canvas, labels] of [[originalCanvasRef.current, null], [overlayCanvasRef.current, mask]]) {
      if (!canvas) continue;
      const slice = composeMriSlice({ shape: geometry.shape, volumePreview, mask: labels, sliceIndex });
      canvas.width = slice.width;
      canvas.height = slice.height;
      const context = canvas.getContext("2d", { alpha: false });
      const image = context.createImageData(slice.width, slice.height);
      image.data.set(slice.data);
      context.putImageData(image, 0, 0);
    }
  }, [geometry, volumePreview, mask, sliceIndex]);

  return (
    <section className="ct-viewer-group" aria-labelledby="mri-viewer-heading">
      <div className="viewer-toolbar">
        <h3 id="mri-viewer-heading">FLAIR and predicted tumor overlay</h3>
        <button type="button" className="icon-button" aria-label="Zoom out MRI image" onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))}>−</button>
        <button type="button" className="icon-button" aria-label="Zoom in MRI image" onClick={() => setZoom((value) => Math.min(3, value + 0.25))}>+</button>
        <button type="button" className="icon-button" aria-label="Reset MRI image zoom" onClick={() => setZoom(1)}>Reset</button>
      </div>
      <p>Native voxel plane, slice {sliceIndex + 1} of {depth}. Tumor core is red, whole-tumor remainder amber, and enhancing tumor magenta. Scroll the panels to pan.</p>
      <label htmlFor="mri-slice">Slice: {sliceIndex + 1} / {depth}</label>
      <input id="mri-slice" type="range" min="0" max={depth - 1} step="1" value={sliceIndex}
        onChange={(event) => setSliceIndex(Number(event.target.value))} />
      <div className="ct-comparison">
        <section className="ct-viewer" aria-label="Original FLAIR slice">
          <h3>Original FLAIR preview</h3>
          <div className="ct-canvas-stage">
            <canvas ref={originalCanvasRef} className="ct-slice-canvas"
              style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
              aria-label={`FLAIR slice ${sliceIndex + 1}, ${width} by ${height}`} />
          </div>
        </section>
        <section className="ct-viewer" aria-label="Predicted tumor segmentation">
          <h3>Predicted segmentation labels</h3>
          <ul className="mri-label-legend">
            <li><span className="mri-label-swatch core" /> Tumor core (label 1)</li>
            <li><span className="mri-label-swatch remainder" /> Whole-tumor remainder (label 2)</li>
            <li><span className="mri-label-swatch enhancing" /> Enhancing tumor (label 4)</li>
          </ul>
          <div className="ct-canvas-stage">
            <canvas ref={overlayCanvasRef} className="ct-slice-canvas"
              style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
              aria-label={`FLAIR with predicted tumor labels, slice ${sliceIndex + 1}`} />
          </div>
        </section>
      </div>
      <dl className="ct-metadata">
        <div><dt>Dimensions</dt><dd>{geometry.shape.join(" × ")}</dd></div>
        <div><dt>Spacing</dt><dd>{geometry.spacing_mm.map((value) => `${value.toFixed(3)} mm`).join(" × ")}</dd></div>
      </dl>
    </section>
  );
}

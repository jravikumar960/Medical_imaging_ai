import { useEffect, useRef, useState } from "react";
import { composeCtSlice } from "./ctSlice.js";

function CtCanvas({ title, geometry, volumePreview, mask, sliceIndex, zoom }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const slice = composeCtSlice({
      shape: geometry.shape,
      volumePreview,
      mask,
      sliceIndex,
    });
    canvas.width = slice.width;
    canvas.height = slice.height;
    const context = canvas.getContext("2d", { alpha: false });
    const image = context.createImageData(slice.width, slice.height);
    image.data.set(slice.data);
    context.putImageData(image, 0, 0);
  }, [geometry, volumePreview, mask, sliceIndex]);

  return (
    <section className="ct-viewer" aria-label={title}>
      <h3>{title}</h3>
      <div className="ct-canvas-stage" aria-label={`${title}, zoom ${Math.round(zoom * 100)} percent`}>
        <canvas ref={canvasRef} className="ct-slice-canvas"
          style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
          aria-label={`${title}, slice ${sliceIndex + 1}`} />
      </div>
    </section>
  );
}

export default function CTSliceViewer({ geometry, volumePreview, mask }) {
  const [sliceIndex, setSliceIndex] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [, , zSize] = geometry.shape;

  useEffect(() => {
    setSliceIndex(0);
    setZoom(1);
  }, [geometry]);

  return (
    <section className="ct-viewer-group" aria-labelledby="ct-viewer-heading">
      <div className="viewer-toolbar">
        <h3 id="ct-viewer-heading">Predicted spleen segmentation</h3>
        <button type="button" className="icon-button" aria-label="Zoom out CT images" onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))}>−</button>
        <button type="button" className="icon-button" aria-label="Zoom in CT images" onClick={() => setZoom((value) => Math.min(3, value + 0.25))}>+</button>
        <button type="button" className="icon-button" aria-label="Reset CT image zoom" onClick={() => setZoom(1)}>Reset</button>
      </div>
      <p>Native voxel plane, slice {sliceIndex + 1} of {zSize}. Red marks the predicted spleen label. Pan using the image-panel scrollbars; this display does not validate the segmentation.</p>
      <label htmlFor="ct-slice">Slice: {sliceIndex + 1} / {zSize}</label>
      <input id="ct-slice" type="range" min="0" max={zSize - 1} step="1" value={sliceIndex}
        onChange={(event) => setSliceIndex(Number(event.target.value))} />
      <div className="ct-comparison">
        <CtCanvas title="Original CT slice" geometry={geometry} volumePreview={volumePreview}
          mask={null} sliceIndex={sliceIndex} zoom={zoom} />
        <CtCanvas title="AI segmentation overlay" geometry={geometry} volumePreview={volumePreview}
          mask={mask} sliceIndex={sliceIndex} zoom={zoom} />
      </div>
      <dl className="ct-metadata">
        <div><dt>Dimensions</dt><dd>{geometry.shape.join(" x ")}</dd></div>
        <div><dt>Spacing</dt><dd>{geometry.spacing_mm.map((value) => `${value.toFixed(3)} mm`).join(" x ")}</dd></div>
        <div><dt>Affine axis codes</dt><dd>{geometry.axis_codes.join(" / ")}</dd></div>
      </dl>
    </section>
  );
}

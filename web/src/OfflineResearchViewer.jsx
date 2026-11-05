import { useEffect, useMemo, useRef, useState } from "react";
import { composeCtResearchSlice } from "./ctSlice.js";

export default function OfflineResearchViewer({
  shape,
  volumePreview,
  predictionMask,
  referenceMask,
  worstSlices = [],
  initialSlice = 0,
}) {
  const canvasRef = useRef(null);
  const depth = Array.isArray(shape) && shape.length === 3 ? shape[2] : 0;
  const [sliceIndex, setSliceIndex] = useState(
    Number.isInteger(initialSlice) && initialSlice >= 0 && initialSlice < depth
      ? initialSlice
      : 0
  );

  const slice = useMemo(() => {
    if (!shape || !volumePreview) return null;
    try {
      return composeCtResearchSlice({
        shape,
        volumePreview,
        predictionMask: predictionMask ?? null,
        referenceMask: referenceMask ?? null,
        sliceIndex,
      });
    } catch (e) {
      return null;
    }
  }, [shape, volumePreview, predictionMask, referenceMask, sliceIndex]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !slice) return;
    canvas.width = slice.width;
    canvas.height = slice.height;
    const ctx = canvas.getContext("2d");
    const img = new ImageData(slice.data, slice.width, slice.height);
    ctx.putImageData(img, 0, 0);
  }, [slice]);

  return (
    <div style={{ display: "flex", gap: "12px" }}>
      <div>
        <canvas ref={canvasRef} style={{ width: "256px", height: "256px", imageRendering: "pixelated" }} />
        <div style={{ marginTop: "8px" }}>
          <button onClick={() => setSliceIndex((i) => Math.max(0, i - 1))} disabled={sliceIndex <= 0}>
            Prev
          </button>
          <span style={{ margin: "0 8px" }}>
            Slice {sliceIndex} / {Math.max(0, depth - 1)}
          </span>
          <button onClick={() => setSliceIndex((i) => Math.min(depth - 1, i + 1))} disabled={sliceIndex >= depth - 1}>
            Next
          </button>
        </div>
        <div style={{ marginTop: "8px", fontSize: "12px" }}>
          <span style={{ color: "#22c55e" }}>■ TP</span>{" "}
          <span style={{ color: "#ef4444" }}>■ FP</span>{" "}
          <span style={{ color: "#3b82f6" }}>■ FN</span>
        </div>
      </div>
      {worstSlices && worstSlices.length > 0 && (
        <div style={{ minWidth: "200px" }}>
          <h4 style={{ margin: "0 0 6px" }}>Worst slices</h4>
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {worstSlices.map((w) => (
              <li key={w.slice_index ?? w} style={{ marginBottom: "4px" }}>
                <button onClick={() => setSliceIndex(w.slice_index ?? w)}>
                  Slice {w.slice_index ?? w} (Dice {Number(w.dice ?? 0).toFixed(3)})
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
"""Render offline research-evaluation slice previews for the 3-case visual review."""
import argparse, base64, json, zlib
from pathlib import Path
import nibabel as nib
import numpy as np


def window_ct(slice2d):
    lo, hi = -57, 164
    x = np.clip(slice2d, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 255).astype("uint8")


# Overlay color constants (RGBA)
PRED_RED = (255, 56, 48)
GT_BLUE = (60, 120, 255)
TP_GREEN = (40, 200, 80)
FP_RED = (255, 56, 48)
FN_BLUE = (60, 120, 255)


def compose_overlay_rgba(ct_slice, pred_slice, gt_slice, mode):
    """Return a (H, W, 4) uint8 RGBA array for one axial slice.

    mode: 'pred' (prediction only, red), 'gt' (ground truth only, blue),
          'compare' (TP green, FP red, FN blue).
    All three input slices must have identical shape. CT windowing is
    identical in every mode. Layout is C-order (width = first axis).
    """
    if not (ct_slice.shape == pred_slice.shape == gt_slice.shape):
        raise ValueError("CT, prediction, and ground-truth slices must have identical shape")
    ct = window_ct(ct_slice)
    rgba = np.zeros(ct.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = ct
    rgba[..., 1] = ct
    rgba[..., 2] = ct
    rgba[..., 3] = 255
    pred = (pred_slice == 1)
    gt = (gt_slice == 1)
    if mode == "pred":
        rgba[pred, 0] = PRED_RED[0]
        rgba[pred, 1] = PRED_RED[1]
        rgba[pred, 2] = PRED_RED[2]
    elif mode == "gt":
        rgba[gt, 0] = GT_BLUE[0]
        rgba[gt, 1] = GT_BLUE[1]
        rgba[gt, 2] = GT_BLUE[2]
    elif mode == "compare":
        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt
        rgba[tp, 0] = TP_GREEN[0]; rgba[tp, 1] = TP_GREEN[1]; rgba[tp, 2] = TP_GREEN[2]
        rgba[fp, 0] = FP_RED[0]; rgba[fp, 1] = FP_RED[1]; rgba[fp, 2] = FP_RED[2]
        rgba[fn, 0] = FN_BLUE[0]; rgba[fn, 1] = FN_BLUE[1]; rgba[fn, 2] = FN_BLUE[2]
    else:
        raise ValueError("unknown overlay mode: %s" % mode)
    return rgba


def compose_slice_rgba(ct_slice, mask_slice):
    """Backward-compatible red-only overlay (prediction mode)."""
    gt = np.zeros_like(mask_slice)
    return compose_overlay_rgba(ct_slice, mask_slice, gt, "pred")


def _check_geometry(img, pred, gt):
    if img.shape != pred.shape or img.shape != gt.shape:
        raise ValueError("shape mismatch: ct%s pred%s gt%s" % (img.shape, pred.shape, gt.shape))
    if not np.allclose(img.affine, pred.affine, atol=1e-5) or not np.allclose(img.affine, gt.affine, atol=1e-5):
        raise ValueError("affine mismatch between CT, prediction, and ground-truth label")


def _load_error_analysis(rd):
    p = rd / "error_analysis.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    out = {}
    for c in data.get("cases", []):
        out[c["case_id"]] = c
    return out



def _load_cases(rd, rep):
    case_payloads = []
    for case in rep["cases"]:
        cid = case["case_id"]
        img = nib.load(case["image_path"])
        pred = nib.load(case["prediction_path"])
        gt = nib.load(case["label_path"])
        _check_geometry(img, pred, gt)
        vol = np.asarray(img.get_fdata())
        pmask = np.asarray(pred.get_fdata()).astype(np.uint8)
        gmask = np.asarray(gt.get_fdata()).astype(np.uint8)
        w, h, d = vol.shape
        err = _load_error_analysis(rd)
        ea = err.get(cid, {})
        slice_stats = {s["slice"]: s for s in ea.get("slice_stats", [])}
        slices = []
        for zi in range(d):
            ct_sl = vol[:, :, zi]
            pr_sl = pmask[:, :, zi]
            gt_sl = gmask[:, :, zi]
            ct_bytes = window_ct(ct_sl).tobytes()
            pr_bytes = (pr_sl == 1).astype(np.uint8).tobytes()
            gt_bytes = (gt_sl == 1).astype(np.uint8).tobytes()
            ss = slice_stats.get(zi, {})
            fp = int(ss.get("fp", int(((pr_sl == 1) & (gt_sl != 1)).sum())))
            fn = int(ss.get("fn", int(((pr_sl != 1) & (gt_sl == 1)).sum())))
            tp = int(((pr_sl == 1) & (gt_sl == 1)).sum())
            slices.append({
                "zi": zi,
                "ct": base64.b64encode(zlib.compress(ct_bytes)).decode("ascii"),
                "pred": base64.b64encode(zlib.compress(pr_bytes)).decode("ascii"),
                "gt": base64.b64encode(zlib.compress(gt_bytes)).decode("ascii"),
                "pred_voxels": int(pr_sl.sum()),
                "gt_voxels": int(gt_sl.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })
        case_payloads.append({
            "case_id": cid,
            "dice": float(case["dice"]),
            "shape": [w, h, d],
            "reference_voxels": int(case.get("reference_voxels", int(gmask.sum()))),
            "slices": slices,
            "worst": {
                "fp": ea.get("worst_fp_slices", []),
                "fn": ea.get("worst_fn_slices", []),
                "dice": ea.get("worst_dice_slices", []),
            },
        })
    return case_payloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.review_dir)
    rep = json.loads((rd / "spleen_evaluation_results.json").read_text())
    case_payloads = _load_cases(rd, rep)

    html = []
    html.append("<!doctype html><html><head><meta charset=\"utf-8\"><title>Research evaluation viewer (offline)</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee;}")
    html.append(".banner{background:#b00020;color:#fff;padding:10px 14px;border-radius:6px;font-weight:700;margin-bottom:16px;}")
    html.append(".case{border:1px solid #444;border-radius:8px;padding:12px;margin:16px 0;}")
    html.append(".controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0;}")
    html.append(".legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0;font-size:13px;}")
    html.append(".legend span{display:inline-flex;align-items:center;gap:4px;}")
    html.append(".dot{width:12px;height:12px;border-radius:2px;display:inline-block;}")
    html.append("canvas{image-rendering:pixelated;max-width:520px;height:auto;border:1px solid #333;background:#000;}")
    html.append("button{background:#333;color:#eee;border:1px solid #555;padding:4px 8px;border-radius:4px;cursor:pointer;}")
    html.append("input[type=range]{width:320px;}")
    html.append("code{color:#9cf;}</style></head><body>")
    html.append("<div class=\"banner\">RESEARCH EVALUATION ONLY — OFFLINE PREDICTIONS. Not a clinical tool. Not validated for patient care. Dice is a training-set metric, not diagnostic accuracy. Reference labels are used only for evaluation/visualization, never as model input.</div>")
    html.append("<h1>Offline spleen segmentation visual review (Phase 5)</h1>")
    html.append("<p>Checkpoint SHA-256: <code>502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe</code> (verified). Cases: lowest/median/highest Dice from 41-case evaluation. Axial slices. CT window fixed [-57, 164] HU in all modes.</p>")

    for ci, cp in enumerate(case_payloads):
        html.append("<section class=\"case\" id=\"case_%d\">" % ci)
        html.append("<h2>%s — Dice %.4f — shape %dx%dx%d — reference voxels %d</h2>" % (cp["case_id"], cp["dice"], cp["shape"][0], cp["shape"][1], cp["shape"][2], cp["reference_voxels"]))
        html.append("<div class=\"legend\">")
        html.append("<span><span class=\"dot\" style=\"background:rgb(255,56,48)\"></span>Prediction (red) / FP (red)</span>")
        html.append("<span><span class=\"dot\" style=\"background:rgb(60,120,255)\"></span>Ground truth (blue) / FN (blue)</span>")
        html.append("<span><span class=\"dot\" style=\"background:rgb(40,200,80)\"></span>True positive (green)</span>")
        html.append("<span><span class=\"dot\" style=\"background:#888\"></span>CT (grayscale)</span>")
        html.append("</div>")
        html.append("<div class=\"controls\">")
        html.append("<label>Mode: <select id=\"mode_%d\"><option value=\"pred\">Prediction only</option><option value=\"gt\">Ground truth only</option><option value=\"compare\">Comparison (TP/FP/FN)</option></select></label>" % ci)
        html.append("<button id=\"prev_%d\">Prev</button>" % ci)
        html.append("<input type=\"range\" id=\"slider_%d\" min=\"0\" max=\"%d\" value=\"0\">" % (ci, cp["shape"][2]-1))
        html.append("<button id=\"next_%d\">Next</button>" % ci)
        html.append("<label>Jump: <select id=\"jump_%d\"></select></label>" % ci)
        html.append("<button id=\"jumpbtn_%d\">Go</button>" % ci)
        html.append("<span id=\"info_%d\"></span>" % ci)
        html.append("</div>")
        html.append("<canvas id=\"canvas_%d\" width=\"%d\" height=\"%d\"></canvas>" % (ci, cp["shape"][0], cp["shape"][1]))
        html.append("</section>")

    html.append("<script>")
    html.append("const CASES = " + json.dumps(case_payloads) + ";")
    html.append("const COLOR = {pred:[255,56,48],gt:[60,120,255],tp:[40,200,80],fp:[255,56,48],fn:[60,120,255]};")
    html.append("function inflate(b64){const bin=atob(b64);const bytes=Uint8Array.from(bin,c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));return new Response(stream).arrayBuffer();}")
    html.append("const cache={};")
    html.append("async function getSlice(ci,zi){if(!cache[ci])cache[ci]={};if(!cache[ci][zi]){const s=CASES[ci].slices[zi];const [ctBuf,prBuf,gtBuf]=await Promise.all([inflate(s.ct),inflate(s.pred),inflate(s.gt)]);cache[ci][zi]={ct:new Uint8Array(ctBuf),pred:new Uint8Array(prBuf),gt:new Uint8Array(gtBuf),meta:s};}return cache[ci][zi];}")
    html.append("function composite(ctx,w,h,sl,mode){const ct=sl.ct,pred=sl.pred,gt=sl.gt;const imgData=ctx.createImageData(w,h);const d=imgData.data;for(let i=0;i<w*h;i++){const v=ct[i];let r=v,g=v,b=v;const p=pred[i]===1;const gg=gt[i]===1;if(mode==='pred'){if(p){r=COLOR.pred[0];g=COLOR.pred[1];b=COLOR.pred[2];}}else if(mode==='gt'){if(gg){r=COLOR.gt[0];g=COLOR.gt[1];b=COLOR.gt[2];}}else if(mode==='compare'){if(p&&gg){r=COLOR.tp[0];g=COLOR.tp[1];b=COLOR.tp[2];}else if(p&&!gg){r=COLOR.fp[0];g=COLOR.fp[1];b=COLOR.fp[2];}else if(!p&&gg){r=COLOR.fn[0];g=COLOR.fn[1];b=COLOR.fn[2];}}const o=i*4;d[o]=r;d[o+1]=g;d[o+2]=b;d[o+3]=255;}ctx.putImageData(imgData,0,0);}")
    html.append("async function renderCase(ci){const cp=CASES[ci];const w=cp.shape[0],h=cp.shape[1];const slider=document.getElementById('slider_'+ci);const zi=parseInt(slider.value,10);const mode=document.getElementById('mode_'+ci).value;const canvas=document.getElementById('canvas_'+ci);const ctx=canvas.getContext('2d');const sl=await getSlice(ci,zi);composite(ctx,w,h,sl,mode);const info=document.getElementById('info_'+ci);info.textContent='slice '+(zi+1)+'/'+cp.shape[2]+'  (pred voxels: '+sl.meta.pred_voxels+', GT voxels: '+sl.meta.gt_voxels+', TP: '+sl.meta.tp+', FP: '+sl.meta.fp+', FN: '+sl.meta.fn+')';}")
    html.append("for(let ci=0;ci<CASES.length;ci++){const slider=document.getElementById('slider_'+ci);const prev=document.getElementById('prev_'+ci);const next=document.getElementById('next_'+ci);const modeSel=document.getElementById('mode_'+ci);const jumpSel=document.getElementById('jump_'+ci);const jumpBtn=document.getElementById('jumpbtn_'+ci);const fillJump=()=>{if(!jumpSel)return;jumpSel.innerHTML='';const w=CASES[ci].worst||{};const add=(label,z)=>{if(Number.isInteger(z)&&z>=0&&z<CASES[ci].shape[2]){const o=document.createElement('option');o.value=String(z);o.textContent=label+' (slice '+(z+1)+')';jumpSel.appendChild(o);}};(w.fp||[]).forEach((z,i)=>add('Worst FP #'+(i+1),z));(w.fn||[]).forEach((z,i)=>add('Worst FN #'+(i+1),z));(w.dice||[]).forEach((z,i)=>add('Worst Dice #'+(i+1),z));if(!jumpSel.options.length){const o=document.createElement('option');o.value='';o.textContent='No worst slices';jumpSel.appendChild(o);}};fillJump();const rerender=()=>renderCase(ci).catch(err=>{document.body.insertAdjacentHTML('beforeend','<pre style=color:#f66>'+err+'</pre>');});slider.addEventListener('input',rerender);prev.addEventListener('click',()=>{slider.value=Math.max(0,parseInt(slider.value,10)-1);rerender();});next.addEventListener('click',()=>{slider.value=Math.min(CASES[ci].shape[2]-1,parseInt(slider.value,10)+1);rerender();});modeSel.addEventListener('change',rerender);if(jumpBtn){jumpBtn.addEventListener('click',()=>{const v=jumpSel.value;if(v!==''){slider.value=v;rerender();}});}rerender();}")
    html.append("</script>")
    html.append("</body></html>")
    out = rd / "research_evaluation_viewer.html"
    out.write_text("\n".join(html), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())

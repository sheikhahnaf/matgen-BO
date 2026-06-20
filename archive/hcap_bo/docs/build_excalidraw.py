"""Build proper .excalidraw scene files for the two flowcharts.

Output:
  docs/figures/flow_original.excalidraw
  docs/figures/flow_v4.excalidraw

Open in excalidraw.com via:  hamburger menu → Open  (or Cmd+O, then drop file)
"""
import json
import random
from pathlib import Path

ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-hcap-bo")
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Excalidraw element factories --------------------------------

def _seed():
    return random.randint(100_000, 9_999_999)


_BASE_DEFAULTS = {
    "angle": 0,
    "strokeStyle": "solid",
    "roughness": 1,
    "opacity": 100,
    "groupIds": [],
    "frameId": None,
    "isDeleted": False,
    "boundElements": [],
    "updated": 1,
    "link": None,
    "locked": False,
    "version": 1,
    "versionNonce": 0,
}


def rect(eid, x, y, w, h, *, fill="#ffffff", stroke="#1e1e1e", stroke_width=2,
         text=None, font_size=18):
    """Rounded rectangle, optionally with bound text label."""
    bound = []
    elements = []
    if text:
        bound.append({"id": f"{eid}_t", "type": "text"})
    elements.append({
        **_BASE_DEFAULTS,
        "type": "rectangle",
        "id": eid,
        "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "roundness": {"type": 3},
        "boundElements": bound,
        "seed": _seed(),
    })
    if text:
        # Bound text — Excalidraw centers it inside the container.
        elements.append({
            **_BASE_DEFAULTS,
            "type": "text",
            "id": f"{eid}_t",
            "x": x + 4, "y": y + h / 2 - font_size,
            "width": w - 8, "height": font_size * (text.count("\n") + 1) * 1.25,
            "strokeColor": "#1e1e1e",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "fontSize": font_size,
            "fontFamily": 2,
            "text": text,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": eid,
            "originalText": text,
            "lineHeight": 1.25,
            "baseline": int(font_size * 0.85),
            "seed": _seed(),
        })
    return elements


def text(eid, x, y, content, *, font_size=18, color="#1e1e1e"):
    return [{
        **_BASE_DEFAULTS,
        "type": "text",
        "id": eid,
        "x": x, "y": y,
        "width": int(len(content) * font_size * 0.55),
        "height": int(font_size * 1.25),
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "fontSize": font_size,
        "fontFamily": 2,
        "text": content,
        "textAlign": "left",
        "verticalAlign": "top",
        "containerId": None,
        "originalText": content,
        "lineHeight": 1.25,
        "baseline": int(font_size * 0.85),
        "seed": _seed(),
    }]


def arrow(eid, x1, y1, x2, y2, *, color="#1e1e1e", stroke_width=2,
          start_id=None, end_id=None, label=None):
    dx, dy = x2 - x1, y2 - y1
    elem = {
        **_BASE_DEFAULTS,
        "type": "arrow",
        "id": eid,
        "x": x1, "y": y1,
        "width": abs(dx), "height": abs(dy),
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": ({"elementId": start_id, "focus": 0, "gap": 1}
                         if start_id else None),
        "endBinding": ({"elementId": end_id, "focus": 0, "gap": 1}
                       if end_id else None),
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "seed": _seed(),
    }
    out = [elem]
    if label:
        # Plain label text near the midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out.extend(text(f"{eid}_lbl", int(mx) + 4, int(my) - 18,
                        label, font_size=14, color=color))
    return out


def write_scene(elements, out_path, source_note):
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": source_note,
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "currentItemFontFamily": 2,
            "gridSize": None,
        },
        "files": {},
    }
    out_path.write_text(json.dumps(scene, indent=2))
    print(f"  wrote {out_path}  ({len(elements)} elements)")


# --------- diagram 1: original MatInvent --------------------------------

def build_original():
    el = []
    el.extend(text("title", 280, -30,
                   "Original MatInvent (Schwaller et al, arXiv 2511.03112, Fig 1a)",
                   font_size=22))
    # Top row
    el.extend(rect("gen", 80, 80, 200, 90, fill="#d0bfff", stroke="#8b5cf6",
                   text="Diffusion Generator\n(RL Agent)", font_size=16))
    el.extend(rect("smp", 340, 80, 200, 90, fill="#a5d8ff", stroke="#4a9eed",
                   text="Generated 3D\nCrystal Structures", font_size=16))
    el.extend(arrow("a1", 280, 125, 340, 125, start_id="gen", end_id="smp"))
    el.extend(rect("opt", 600, 80, 220, 90, fill="#fff3bf", stroke="#f59e0b",
                   text="Geometry Optimization\n(MLIP)", font_size=16))
    el.extend(arrow("a2", 540, 125, 600, 125, start_id="smp", end_id="opt"))
    el.extend(rect("sun", 860, 80, 230, 90, fill="#ffd8a8", stroke="#f59e0b",
                   text="SUN Filter\n(Valid · Stable\nUnique · Novel)", font_size=14))
    el.extend(arrow("a3", 820, 125, 860, 125, start_id="opt", end_id="sun"))
    # Down to property eval
    el.extend(rect("prop", 860, 280, 230, 100, fill="#ffc9c9", stroke="#ef4444",
                   text="Property Eval\n(eSEN + phonopy / EOS)\nEXPENSIVE ORACLE",
                   font_size=13))
    el.extend(arrow("a4", 975, 170, 975, 280,
                    color="#ef4444", start_id="sun", end_id="prop"))
    # Reward
    el.extend(rect("rew", 600, 280, 220, 100, fill="#fff3bf", stroke="#f59e0b",
                   text="Reward Assignment", font_size=16))
    el.extend(arrow("a5", 860, 330, 820, 330, start_id="prop", end_id="rew"))
    # Fine-tune — placed so its x-range overlaps the Generator box (80–280)
    # so the loop arrow lands cleanly on Diffusion Generator instead of "Generated Structures".
    el.extend(rect("ft", 80, 280, 380, 100, fill="#b2f2bb", stroke="#22c55e",
                   text="REINFORCE Fine-tune\n+ KL reg + Replay + Diversity Filter",
                   font_size=13))
    el.extend(arrow("a6", 600, 330, 460, 330, start_id="rew", end_id="ft"))
    # Loop back: REINFORCE top → Diffusion Generator bottom (single straight up arrow)
    el.extend(arrow("loop", 180, 280, 180, 170, color="#22c55e",
                    stroke_width=3, start_id="ft", end_id="gen", label="policy update"))

    el.extend(text("note1", 80, 420,
                   "Bottleneck: every SUN-filter survivor (~5–15 per cycle) goes through",
                   font_size=14, color="#757575"))
    el.extend(text("note2", 80, 444,
                   "a full eSEN+phonopy / EOS oracle (~1–3 min on T4 GPU).",
                   font_size=14, color="#757575"))
    el.extend(text("note3", 80, 468,
                   "20-cycle run = 50–300 oracle calls; entire run = 1–5 GPU-hours.",
                   font_size=14, color="#757575"))

    write_scene(el, OUT_DIR / "flow_original.excalidraw",
                "matinvent-hcap-bo / phase3-v4-bm — Original MatInvent")


# --------- diagram 2: v4 BO+RL extension --------------------------------

def build_v4():
    el = []
    el.extend(text("title", 280, -30,
                   "Our v4 Extension: BO+RL with GP-routed selection",
                   font_size=22))

    # top row — generator → filter → featurize
    el.extend(rect("gen", 80, 60, 200, 90, fill="#d0bfff", stroke="#8b5cf6",
                   text="Diffusion Generator\n(RL Agent)", font_size=16))
    el.extend(rect("smp", 340, 60, 200, 90, fill="#a5d8ff", stroke="#4a9eed",
                   text="Generated\nStructures", font_size=16))
    el.extend(arrow("a1", 280, 105, 340, 105, start_id="gen", end_id="smp"))
    el.extend(rect("opt", 600, 60, 200, 90, fill="#fff3bf", stroke="#f59e0b",
                   text="GeoOpt + SUN Filter", font_size=16))
    el.extend(arrow("a2", 540, 105, 600, 105, start_id="smp", end_id="opt"))
    el.extend(rect("feat", 860, 60, 220, 90, fill="#c3fae8", stroke="#06b6d4",
                   text="ORB-PCA50 Featurize", font_size=16))
    el.extend(arrow("a3", 800, 105, 860, 105, start_id="opt", end_id="feat"))

    # BO band
    el.extend([{
        **_BASE_DEFAULTS,
        "type": "rectangle",
        "id": "newzone",
        "x": 60, "y": 200, "width": 1280, "height": 280,
        "strokeColor": "#f59e0b",
        "backgroundColor": "#fff3bf",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "roundness": {"type": 3},
        "opacity": 25,
        "seed": _seed(),
    }])
    el.extend(text("newlbl", 360, 210,
                   "NEW: Bayesian Optimization layer (LocalESEN_*_GPRoutedV4)",
                   font_size=15, color="#a16207"))

    # GP, warm-start, acquisition, top-K
    el.extend(rect("gp", 860, 260, 220, 100, fill="#eebefa", stroke="#ec4899",
                   text="GP Surrogate\nFixedNoiseGP\n(μ, σ predictions)", font_size=13))
    el.extend(arrow("a4", 970, 150, 970, 260,
                    start_id="feat", end_id="gp"))
    el.extend(rect("ws", 1100, 260, 200, 100, fill="#a5d8ff", stroke="#4a9eed",
                   text="Warm-start\n446 prior labels\n(decay)", font_size=13))
    el.extend(arrow("a5", 1100, 310, 1080, 310,
                    color="#4a9eed", start_id="ws", end_id="gp"))
    el.extend(rect("acq", 540, 260, 240, 100, fill="#d0bfff", stroke="#8b5cf6",
                   text="Acquisition\nEI(μ, σ, f_best)\n+ DPP diversity", font_size=13))
    el.extend(arrow("a6", 860, 310, 780, 310, start_id="gp", end_id="acq"))
    el.extend(rect("topk", 240, 260, 240, 100, fill="#b2f2bb", stroke="#22c55e",
                   text="Select top-K=4\n(min K, ⌈0.5N⌉)", font_size=13))
    el.extend(arrow("a7", 540, 310, 480, 310, start_id="acq", end_id="topk"))

    # Oracle + honest reward
    el.extend(rect("oracle", 240, 400, 240, 70, fill="#ffc9c9", stroke="#ef4444",
                   text="Oracle: top-K only\n(eSEN+phonon/EOS)", font_size=12))
    el.extend(arrow("a8", 360, 360, 360, 400,
                    color="#ef4444", start_id="topk", end_id="oracle"))
    el.extend(rect("mask", 540, 400, 280, 70, fill="#ffd8a8", stroke="#f59e0b",
                   text="Honest reward (NaN for non-oracle\n→ zero RL gradient)", font_size=12))
    el.extend(arrow("a9", 480, 435, 540, 435, start_id="oracle", end_id="mask"))

    # REINFORCE — anchored under Diffusion Generator (x range 80-280) so loop
    # arrow lands cleanly on Diffusion Generator bottom.
    el.extend(rect("ft", 80, 540, 200, 90, fill="#b2f2bb", stroke="#22c55e",
                   text="REINFORCE\n+ KL reg + Replay", font_size=14))
    el.extend(arrow("a10", 480, 470, 280, 580,
                    start_id="oracle", end_id="ft"))
    # Loop back: REINFORCE top → Diffusion Generator bottom (single straight arrow)
    el.extend(arrow("loop", 180, 540, 180, 150,
                    color="#22c55e", stroke_width=3,
                    start_id="ft", end_id="gen", label="policy update"))

    el.extend(text("k1", 80, 660, "v4 knobs:", font_size=14))
    el.extend(text("k2", 80, 685,
                   "GP_TOP_K=4   GP_K_RATIO=0.5   GP_ANCHOR_EVERY=999 (off)   GP_COLD_START_MIN=0",
                   font_size=12, color="#757575"))
    el.extend(text("k3", 80, 710,
                   "GP_DPP_LAMBDA=0.5   GP_WARMSTART_DECAY=0.9   GP_ACQUISITION=ei",
                   font_size=12, color="#757575"))

    write_scene(el, OUT_DIR / "flow_v4.excalidraw",
                "matinvent-hcap-bo / phase3-v4-bm — v4 BO+RL")


if __name__ == "__main__":
    random.seed(42)  # deterministic seeds
    build_original()
    build_v4()
    print("\nDone. To edit:")
    print("  1. Open https://excalidraw.com")
    print("  2. Hamburger menu (top-left) → Open")
    print("  3. Select docs/figures/flow_original.excalidraw or flow_v4.excalidraw")

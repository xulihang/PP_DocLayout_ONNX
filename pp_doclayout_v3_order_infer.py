"""
Test PP-DocLayoutV3 with reading order recovery.
Uses the HuggingFace-exported ONNX model from:
  https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX

Usage:
  1. Download the model files (both .onnx and .onnx.data must be in same dir):
     python pp_doclayout_v3_order_infer.py <image_path> --model PP-DocLayoutV3.onnx

  2. Or auto-download from HuggingFace (requires huggingface_hub):
     pip install huggingface_hub
     python pp_doclayout_v3_order_infer.py <image_path>
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ── Model config ──────────────────────────────────────────────────────────
INPUT_SIZE = 800
NUM_QUERIES = 300
NUM_CLASSES = 25
CONF_THRESHOLD = 0.3

# Labels from PaddlePaddle/PP-DocLayoutV3 (25 classes)
ID2LABEL = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "document_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

# Colors for visualization (BGR)
COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 128), (0, 255, 0),
    (255, 0, 0), (255, 0, 128), (255, 128, 0), (255, 255, 0),
    (128, 0, 255), (128, 255, 0), (0, 128, 128), (128, 0, 128),
    (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 128, 255),
    (128, 255, 128), (255, 128, 128), (255, 128, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (255, 255, 128), (255, 128, 64),
    (64, 128, 255),
]


# ── Download model ────────────────────────────────────────────────────────
def download_model(target_dir="."):
    """Download the ONNX model from HuggingFace."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = target_dir / "PP-DocLayoutV3.onnx"
    data_path = target_dir / "PP-DocLayoutV3.onnx.data"

    if onnx_path.exists() and data_path.exists():
        return str(onnx_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        print(f"Or manually download to: {target_dir.absolute()}")
        print("  Files: PP-DocLayoutV3.onnx  +  PP-DocLayoutV3.onnx.data")
        print("  From:  https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX")
        sys.exit(1)

    print("Downloading PP-DocLayoutV3.onnx...")
    hf_hub_download(
        repo_id="Bei0001/PP-DocLayoutV3-ONNX",
        filename="PP-DocLayoutV3.onnx",
        local_dir=str(target_dir),
    )
    print("Downloading PP-DocLayoutV3.onnx.data...")
    hf_hub_download(
        repo_id="Bei0001/PP-DocLayoutV3-ONNX",
        filename="PP-DocLayoutV3.onnx.data",
        local_dir=str(target_dir),
    )
    return str(onnx_path)


# ── Load model ────────────────────────────────────────────────────────────
def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = model_path + ".data"
    if not Path(data_path).exists():
        print(f"Warning: .onnx.data sidecar not found at {data_path}")
        print("The model may fail to load without it.")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=opts)
    return sess


# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess(image_rgb):
    """
    Resize to 800x800, normalize.
    mean=[0,0,0], std=[1,1,1] (so just /255).
    """
    ori_h, ori_w = image_rgb.shape[:2]

    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # Normalize: uint8 -> float32, scale 1/255
    img = resized.astype(np.float32) / 255.0

    # HWC -> CHW -> add batch dim
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, ori_w, ori_h


# ── Inference ─────────────────────────────────────────────────────────────
def inference(session, pixel_values):
    outputs = session.run(
        ["logits", "pred_boxes", "order_logits"],
        {"pixel_values": pixel_values},
    )
    return outputs  # logits, pred_boxes, order_logits


# ── Postprocessing ────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_reading_order(order_logits, valid_indices):
    """
    Decode reading order from order_logits (300, 300) for valid queries.

    order_logits[i, j] = pairwise comparison logit: how likely query i
    comes BEFORE query j. Matches the HuggingFace transformers implementation
    of _get_order_seqs() in PPDocLayoutV3ImageProcessor.

    Algorithm (pairwise voting):
      1. sigmoid(order_logits) -> scores[i,j] = P(i before j)
      2. votes[i] = sum_{j>i} P(i before j) + sum_{j<i} (1 - P(j before i))
      3. sort queries by votes (higher = earlier in reading order)
    """
    N = len(valid_indices)
    if N == 0:
        return np.array([], dtype=np.int32)

    # Extract sub-matrix for valid queries
    sub_logits = order_logits[valid_indices][:, valid_indices]  # (N, N)

    # Sigmoid: P(i before j)
    order_scores = 1.0 / (1.0 + np.exp(-sub_logits))  # (N, N)

    # Upper triangle: sum of P(i before j) for all j > i
    upper = np.triu(order_scores, k=1).sum(axis=1)  # (N,)

    # Lower triangle inverted: transposed scores in lower triangle
    # order_scores[j,i] = P(j before i), so 1 - P(j before i) = P(i before j)
    lower_inverted = np.tril(1.0 - order_scores.T, k=-1).sum(axis=1)  # (N,)

    votes = upper + lower_inverted  # (N,)

    # Higher votes = earlier in reading order
    order_pointers = np.argsort(-votes)  # descending
    order_seq = np.zeros(N, dtype=np.int32)
    for rank, idx in enumerate(order_pointers):
        order_seq[idx] = rank

    return order_seq


def refine_order_by_layout(results):
    """
    Refine reading order using spatial layout (top-to-bottom, left-to-right).
    The model's pairwise voting gives a rough order; this corrects it with
    geometric constraints that match how humans read documents.
    """
    N = len(results)
    if N <= 1:
        return

    # Compute average box height for line grouping tolerance
    heights = [r["bbox"][3] - r["bbox"][1] for r in results]
    avg_h = sum(heights) / N
    line_tolerance = avg_h * 0.5

    def sort_key(r):
        bbox = r["bbox"]
        y_center = (bbox[1] + bbox[3]) / 2.0
        x_center = (bbox[0] + bbox[2]) / 2.0
        # Quantize y_center into "lines" using line_tolerance
        line_idx = round(y_center / line_tolerance)
        return (line_idx, x_center)

    results.sort(key=sort_key)

    # Reassign sequential order numbers
    for i, r in enumerate(results):
        r["order"] = i


def postprocess(logits, pred_boxes, order_logits, ori_w, ori_h, conf_thres=0.3):
    """
    Decode model outputs into detection results with reading order.

    Returns list of dicts: {label, score, bbox, order}
    """
    # logits: (1, 300, 25) -> scores via sigmoid (matches HF original: BCE per class)
    logits = logits[0]  # (300, 25)
    # Use sigmoid (not softmax): PP-DocLayoutV3 is DETR-based, trained with
    # sigmoid + binary cross-entropy — each class is independent.
    scores_per_class = sigmoid(logits)  # (300, 25)
    scores = np.max(scores_per_class, axis=1)  # (300,)
    class_ids = np.argmax(scores_per_class, axis=1)  # (300,)

    # pred_boxes: (1, 300, 4) in (cx, cy, w, h) normalized to [0,1]
    boxes = pred_boxes[0]  # (300, 4)

    # order_logits: (1, 300, 300)
    order_logits = order_logits[0]  # (300, 300)

    # Filter by confidence
    keep = np.where(scores >= conf_thres)[0]
    if len(keep) == 0:
        return []

    # 1. Decode reading order FIRST (on all kept queries, before NMS)
    order_seq = decode_reading_order(order_logits, keep)

    # 2. Build result list
    results = []
    for idx, (i, ord_pos) in enumerate(zip(keep, order_seq)):
        cx, cy, w, h = boxes[i]

        # Convert (cx, cy, w, h) normalized -> (x1, y1, x2, y2) absolute pixels
        x1 = (cx - w / 2) * ori_w
        y1 = (cy - h / 2) * ori_h
        x2 = (cx + w / 2) * ori_w
        y2 = (cy + h / 2) * ori_h

        # Clip to image bounds
        x1 = max(0, min(int(x1), ori_w))
        y1 = max(0, min(int(y1), ori_h))
        x2 = max(0, min(int(x2), ori_w))
        y2 = max(0, min(int(y2), ori_h))

        cls_id = int(class_ids[i])
        label = ID2LABEL.get(cls_id, f"cls_{cls_id}")

        results.append({
            "label": label,
            "cls_id": cls_id,
            "score": float(scores[i]),
            "bbox": [x1, y1, x2, y2],
            "order": int(ord_pos),
        })

    # 3. Refine reading order with spatial layout (no NMS — DETR queries are naturally non-overlapping)
    refine_order_by_layout(results)

    # 4. Sort by reading order
    results.sort(key=lambda r: r["order"])
    return results


# ── Visualization ─────────────────────────────────────────────────────────
def draw_results(image_rgb, results):
    """Draw bounding boxes with reading order numbers."""
    result = image_rgb.copy()

    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        score = det["score"]
        order = det["order"]
        cls_id = det["cls_id"]

        color = COLORS[cls_id % len(COLORS)]

        thickness = max(2, min(result.shape[0], result.shape[1]) // 400)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Text with reading order number
        text = f"#{order} {label} {score:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(result.shape[0], result.shape[1]) / 1400)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = y1
        cv2.rectangle(result, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)

        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result, text, (x1 + 2, label_y2 - baseline - 2),
                    font, font_scale, text_color, 1, cv2.LINE_AA)

    return result


# ── Main ──────────────────────────────────────────────────────────────────
def detect(image_path, model_path=None, conf_thres=0.3, output_path=None, auto_download=True):
    """Run layout detection with reading order on an image."""
    # Load or download model
    if model_path is None:
        model_path = "PP-DocLayoutV3.onnx"

    if not Path(model_path).exists():
        if auto_download:
            model_path = download_model(Path(model_path).parent or ".")
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download from: https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX\n"
                "Files needed: PP-DocLayoutV3.onnx + PP-DocLayoutV3.onnx.data"
            )

    sess = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"  Input:  {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}")
    for o in sess.get_outputs():
        print(f"  Output: {o.name} {o.shape}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Preprocess
    pixel_values, ori_w, ori_h = preprocess(image_rgb)

    # Inference
    logits, pred_boxes, order_logits = inference(sess, pixel_values)

    # Postprocess
    results = postprocess(logits, pred_boxes, order_logits, ori_w, ori_h, conf_thres)

    print(f"\nDetections: {len(results)}")
    print(f"{'Order':>5} {'Label':<20} {'Score':>7} {'BBox'}")
    print("-" * 70)
    for det in results:
        print(f"{det['order']:>5} {det['label']:<20} {det['score']:>6.1%} "
              f"({det['bbox'][0]},{det['bbox'][1]})-({det['bbox'][2]},{det['bbox'][3]})")

    # Draw and save
    annotated = draw_results(image_rgb, results)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    if output_path is None:
        output_path = f"{Path(image_path).stem}_layout_order.jpg"
    cv2.imwrite(output_path, annotated_bgr)
    print(f"\nResult saved to: {output_path}")

    return results, annotated_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PP-DocLayoutV3 layout detection with reading order recovery"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model", default=None,
        help="Path to PP-DocLayoutV3.onnx (auto-download if missing)"
    )
    parser.add_argument("--output", "-o", default=None, help="Path to output image")
    parser.add_argument(
        "--conf", "-c", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Don't auto-download model"
    )
    args = parser.parse_args()

    detect(
        args.image,
        model_path=args.model,
        conf_thres=args.conf,
        output_path=args.output,
        auto_download=not args.no_download,
    )

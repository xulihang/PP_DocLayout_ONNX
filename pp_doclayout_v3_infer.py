"""
Standalone layout detection script using pp_doc_layoutv3.onnx.

Usage:
    python detect_layout.py <image_path> [--output <output_path>] [--conf <threshold>]

Example:
    python detect_layout.py test.jpg
    python detect_layout.py test.jpg --output result.jpg --conf 0.5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path(__file__).parent / "pp_doc_layoutv3.onnx"
INPUT_SIZE = 800


def load_model(model_path):
    """Load ONNX model and return session, input names, labels."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 4
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options=sess_opt)

    input_names = [v.name for v in session.get_inputs()]

    meta = session.get_modelmeta().custom_metadata_map
    labels = meta.get("character", "").splitlines()
    if not labels:
        print("Warning: No 'character' metadata found, using index-based labels.")

    return session, input_names, labels


def preprocess(image):
    """Resize to 800x800, normalize, and prepare model inputs.

    The model has 3 inputs:
      - im_shape:  [[h, w]] of the resized image (model input size), float32
      - image:     (1, 3, 800, 800) normalized, float32
      - scale_factor: [[h_ratio, w_ratio]] = [[new_h/ori_h, new_w/ori_w]], float32
    """
    ori_h, ori_w = image.shape[:2]

    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # Normalize: split channels, float32, scale by 1/255
    b, g, r = cv2.split(resized)
    img = cv2.merge([
        b.astype(np.float32) / 255.0,
        g.astype(np.float32) / 255.0,
        r.astype(np.float32) / 255.0,
    ])

    # HWC -> CHW, then add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    im_shape = np.array([[float(INPUT_SIZE), float(INPUT_SIZE)]], dtype=np.float32)
    scale_factor = np.array([[INPUT_SIZE / ori_h, INPUT_SIZE / ori_w]], dtype=np.float32)

    return {
        "im_shape": im_shape,
        "image": img,
        "scale_factor": scale_factor,
    }, ori_w, ori_h


def inference(session, input_names, feeds):
    """Run ONNX inference. Returns raw model outputs."""
    input_feed = {name: feeds[name] for name in input_names}
    return session.run(None, input_feed)


def parse_outputs(outputs):
    """Extract valid boxes from model outputs.

    outputs[0]: (M, 6) array of [cls_id, score, x1, y1, x2, y2]
    outputs[1]: (1,) array with count of valid boxes
    """
    boxes_all = outputs[0]
    num_valid = int(outputs[1][0])
    if num_valid <= 0 or len(boxes_all) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return boxes_all[:num_valid].copy()


def nms(boxes, iou_threshold=0.6):
    """Per-class NMS. boxes: (N, 6) [cls_id, score, x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return boxes

    order = np.argsort(boxes[:, 1])[::-1]
    boxes = boxes[order]

    keep = []
    while len(boxes) > 0:
        best = boxes[0]
        keep.append(best)
        boxes = boxes[1:]
        if len(boxes) == 0:
            break

        xx1 = np.maximum(best[2], boxes[:, 2])
        yy1 = np.maximum(best[3], boxes[:, 3])
        xx2 = np.minimum(best[4], boxes[:, 4])
        yy2 = np.minimum(best[5], boxes[:, 5])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_best = max(0, (best[4] - best[2]) * (best[5] - best[3]))
        areas = np.maximum(0, (boxes[:, 4] - boxes[:, 2]) * (boxes[:, 5] - boxes[:, 3]))
        iou = inter / (area_best + areas - inter + 1e-6)

        same_cls = boxes[:, 0] == best[0]
        mask = ~(((same_cls) & (iou > iou_threshold)) | ((~same_cls) & (iou > 0.98)))
        boxes = boxes[mask]

    return np.array(keep)


def postprocess(boxes, conf_thres, ori_w, ori_h):
    """Filter by confidence, NMS, clip to image bounds."""
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = boxes[boxes[:, 1] >= conf_thres]
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes[:, 2:] = np.round(boxes[:, 2:]).astype(np.float32)

    boxes = nms(boxes)

    # Clip to image bounds (boxes are already in original image coords)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ori_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, ori_h)
    boxes[:, 4] = np.clip(boxes[:, 4], 0, ori_w)
    boxes[:, 5] = np.clip(boxes[:, 5], 0, ori_h)

    boxes[:, 2:] = boxes[:, 2:].astype(int)
    return boxes


def get_class_colors(num_classes):
    """Generate visually distinct colors for each class."""
    rng = np.random.RandomState(42)
    return rng.randint(80, 255, (num_classes, 3), dtype=np.uint8)


def draw_detections(image, boxes, labels):
    """Draw bounding boxes and labels on the image."""
    colors = get_class_colors(len(labels))
    result = image.copy()

    for box in boxes:
        cls_id = int(box[0])
        score = box[1]
        x1, y1, x2, y2 = int(box[2]), int(box[3]), int(box[4]), int(box[5])

        label_name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
        color = colors[cls_id % len(colors)].tolist()

        thickness = max(2, min(result.shape[0], result.shape[1]) // 400)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        text = f"{label_name} {score:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(result.shape[0], result.shape[1]) / 1200)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = y1
        cv2.rectangle(result, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)

        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result, text, (x1 + 2, label_y2 - baseline - 2),
                    font, font_scale, text_color, 1, cv2.LINE_AA)

    return result


def detect(image_path, model_path=None, conf_thres=0.3, output_path=None):
    """Run layout detection and save annotated result."""
    if model_path is None:
        model_path = MODEL_PATH

    session, input_names, labels = load_model(model_path)
    print(f"Model loaded. {len(labels)} classes.")
    for inp in session.get_inputs():
        print(f"  Input: {inp.name} {inp.shape}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Preprocess
    feeds, ori_w, ori_h = preprocess(image_rgb)

    # Inference
    outputs = inference(session, input_names, feeds)
    raw_boxes = parse_outputs(outputs)
    print(f"Raw detections: {len(raw_boxes)}")

    # Postprocess
    boxes = postprocess(raw_boxes, conf_thres, ori_w, ori_h)
    print(f"After filtering: {len(boxes)} detections")

    # Print results
    for box in boxes:
        cls_id = int(box[0])
        label_name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
        print(f"  {label_name}: {box[1]:.1%} @ ({box[2]},{box[3]})-({box[4]},{box[5]})")

    # Draw and save
    annotated = draw_detections(image_rgb, boxes, labels)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    if output_path is None:
        output_path = f"{Path(image_path).stem}_layout.jpg"
    cv2.imwrite(output_path, annotated_bgr)
    print(f"Result saved to: {output_path}")

    return boxes, annotated_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layout detection using pp_doc_layoutv3.onnx"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model", default=None,
        help="Path to ONNX model (default: pp_doc_layoutv3.onnx in script dir)"
    )
    parser.add_argument("--output", "-o", default=None, help="Path to output image")
    parser.add_argument(
        "--conf", "-c", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)"
    )
    args = parser.parse_args()

    detect(args.image, model_path=args.model, conf_thres=args.conf, output_path=args.output)

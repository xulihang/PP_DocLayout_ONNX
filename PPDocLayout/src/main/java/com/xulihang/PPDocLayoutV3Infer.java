package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.*;

/**
 * Layout detection with reading order using PP-DocLayoutV3 ONNX model
 * (HuggingFace Transformers export).
 *
 * Model: https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX
 * Needs both PP-DocLayoutV3.onnx and PP-DocLayoutV3.onnx.data in the same directory.
 *
 * Usage:
 *   PPDocLayoutV3Infer detector = new PPDocLayoutV3Infer("PP-DocLayoutV3.onnx");
 *   List&lt;DetectionResult&gt; results = detector.detect(image, 0.3f);
 *   // DetectionResult: category, confidence, bbox=[x1,y1,x2,y2], order
 */
public class PPDocLayoutV3Infer {

    private OrtSession session;
    private static final int INPUT_SIZE = 800;
    private static final int NUM_CLASSES = 25;

    // ── 25-class labels from PaddlePaddle/PP-DocLayoutV3 ─────────────────
    private static final String[] LABELS = {
        "abstract", "algorithm", "aside_text", "chart", "content",
        "display_formula", "document_title", "figure_title", "footer", "footer_image",
        "footnote", "formula_number", "header", "header_image", "image",
        "inline_formula", "number", "paragraph_title", "reference", "reference_content",
        "seal", "table", "text", "vertical_text", "vision_footnote",
    };

    private static final Scalar[] COLORS;

    static {
        Random rng = new Random(42);
        COLORS = new Scalar[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++) {
            COLORS[i] = new Scalar(
                    rng.nextInt(176) + 80,
                    rng.nextInt(176) + 80,
                    rng.nextInt(176) + 80
            );
        }
    }

    // ── Constructor ─────────────────────────────────────────────────────

    public PPDocLayoutV3Infer(String modelPath) throws OrtException {
        // Verify .data sidecar exists
        String dataPath = modelPath + ".data";
        if (!java.nio.file.Files.exists(Paths.get(dataPath))) {
            System.out.println("Warning: .onnx.data sidecar not found at " + dataPath);
            System.out.println("The model may fail to load. Make sure both files are in the same directory.");
            System.out.println("Download from: https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX");
        }

        OrtSession.SessionOptions sessOpt = new OrtSession.SessionOptions();
        sessOpt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, sessOpt);

        System.out.println("Model loaded. Inputs:");
        for (NodeInfo info : session.getInputInfo().values()) {
            System.out.println("  " + info.getName() + " " + java.util.Arrays.toString(((TensorInfo) info.getInfo()).getShape()));
        }
        System.out.println("Outputs:");
        for (NodeInfo info : session.getOutputInfo().values()) {
            System.out.println("  " + info.getName() + " " + java.util.Arrays.toString(((TensorInfo) info.getInfo()).getShape()));
        }
        System.out.println(LABELS.length + " classes.");
    }

    // ── Preprocessing ───────────────────────────────────────────────────

    private float[] preprocess(Mat imageRgb) {
        Mat resized = new Mat();
        Imgproc.resize(imageRgb, resized, new Size(INPUT_SIZE, INPUT_SIZE), 0, 0, Imgproc.INTER_CUBIC);

        // Convert to float32, normalize by 1/255 (mean=[0,0,0], std=[1,1,1])
        resized.convertTo(resized, CvType.CV_32F, 1.0 / 255.0);

        // Split RGB channels -> CHW
        List<Mat> channels = new ArrayList<>(3);
        Core.split(resized, channels);

        float[] inputData = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int idx = 0;
        for (Mat channel : channels) {
            float[] channelData = new float[(int) channel.total()];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, inputData, idx, channelData.length);
            idx += channelData.length;
        }

        return inputData;
    }

    // ── Inference ───────────────────────────────────────────────────────

    private OrtSession.Result runInference(float[] pixelValues) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(pixelValues),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("pixel_values", inputTensor);

        OrtSession.Result outputs = session.run(inputs);
        inputTensor.close();
        return outputs;
    }

    // ── Reading order decoding (pairwise voting) ────────────────────────

    /**
     * Decode reading order from order_logits (N, N) for valid query indices.
     *
     * order_logits[i,j] = pairwise logit: how likely query i comes BEFORE query j.
     * Algorithm matches HuggingFace PPDocLayoutV3ImageProcessor._get_order_seqs():
     *   1. sigmoid -> scores[i,j] = P(i before j)
     *   2. votes[i] = sum_{j>i} P(i before j) + sum_{j<i} (1 - P(j before i))
     *   3. sort by votes descending = reading order
     */
    private int[] decodeReadingOrder(float[][] orderLogits, int[] validIndices) {
        int N = validIndices.length;
        if (N == 0) return new int[0];

        // Extract sub-matrix for valid queries
        double[][] scores = new double[N][N];
        for (int a = 0; a < N; a++) {
            for (int b = 0; b < N; b++) {
                // sigmoid
                double x = orderLogits[validIndices[a]][validIndices[b]];
                scores[a][b] = 1.0 / (1.0 + Math.exp(-x));
            }
        }

        // Compute votes
        double[] votes = new double[N];
        for (int i = 0; i < N; i++) {
            // Upper triangle: sum P(i before j) for j > i
            double upper = 0;
            for (int j = i + 1; j < N; j++) {
                upper += scores[i][j];
            }
            // Lower triangle inverted: sum (1 - P(j before i)) for j < i
            double lower = 0;
            for (int j = 0; j < i; j++) {
                lower += (1.0 - scores[j][i]);
            }
            votes[i] = upper + lower;
        }

        // Sort by votes descending -> order pointers
        Integer[] pointers = new Integer[N];
        for (int i = 0; i < N; i++) pointers[i] = i;
        Arrays.sort(pointers, (a, b) -> Double.compare(votes[b], votes[a]));

        int[] orderSeq = new int[N];
        for (int rank = 0; rank < N; rank++) {
            orderSeq[pointers[rank]] = rank;
        }
        return orderSeq;
    }

    // ── NMS on raw query data ───────────────────────────────────────────

    /**
     * NMS on raw query boxes before order decoding.
     * Returns indices of surviving queries (indices into the filtered keep array).
     */
    private int[] nmsOnQueries(float[][] absBoxes, float[] scores, int[] clsIds, float iouThresh) {
        int n = absBoxes.length;
        if (n == 0) return new int[0];

        // Sort by score descending
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));

        boolean[] suppressed = new boolean[n];
        int keepCount = 0;

        for (int idx = 0; idx < n; idx++) {
            int best = order[idx];
            if (suppressed[best]) continue;

            for (int j = idx + 1; j < n; j++) {
                int other = order[j];
                if (suppressed[other]) continue;
                if (clsIds[best] != clsIds[other]) continue;

                float iou = computeIoU(absBoxes[best], absBoxes[other]);
                if (iou >= iouThresh) {
                    suppressed[other] = true;
                }
            }
        }

        int[] kept = new int[n];
        for (int i = 0; i < n; i++) {
            if (!suppressed[i]) kept[keepCount++] = i;
        }
        return Arrays.copyOf(kept, keepCount);
    }

    private float computeIoU(float[] a, float[] b) {
        float xx1 = Math.max(a[0], b[0]);
        float yy1 = Math.max(a[1], b[1]);
        float xx2 = Math.min(a[2], b[2]);
        float yy2 = Math.min(a[3], b[3]);
        float inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
        float areaA = Math.max(0, (a[2] - a[0]) * (a[3] - a[1]));
        float areaB = Math.max(0, (b[2] - b[0]) * (b[3] - b[1]));
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    // ── Postprocessing ──────────────────────────────────────────────────

    private List<DetectionResult> postprocess(
            float[][] logits, float[][] predBoxes, float[][] orderLogits,
            int oriW, int oriH, float confThres) {

        // Compute class scores via softmax
        int n = logits.length;
        float[] scores = new float[n];
        int[] classIds = new int[n];
        for (int i = 0; i < n; i++) {
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < NUM_CLASSES; j++) {
                if (logits[i][j] > maxLogit) maxLogit = logits[i][j];
            }
            double sumExp = 0;
            double[] exps = new double[NUM_CLASSES];
            for (int j = 0; j < NUM_CLASSES; j++) {
                exps[j] = Math.exp(logits[i][j] - maxLogit);
                sumExp += exps[j];
            }
            float bestProb = 0;
            int bestCls = 0;
            for (int j = 0; j < NUM_CLASSES; j++) {
                float prob = (float) (exps[j] / sumExp);
                if (prob > bestProb) {
                    bestProb = prob;
                    bestCls = j;
                }
            }
            scores[i] = bestProb;
            classIds[i] = bestCls;
        }

        // ── 1. Filter by confidence ──
        List<Integer> keepList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (scores[i] >= confThres) keepList.add(i);
        }
        if (keepList.isEmpty()) return new ArrayList<>();

        int[] keep = keepList.stream().mapToInt(i -> i).toArray();
        int K = keep.length;

        // Build filtered arrays
        float[] keepScores = new float[K];
        int[] keepClsIds = new int[K];
        float[][] absBoxes = new float[K][4];
        for (int k = 0; k < K; k++) {
            int i = keep[k];
            keepScores[k] = scores[i];
            keepClsIds[k] = classIds[i];
            float cx = predBoxes[i][0], cy = predBoxes[i][1];
            float w = predBoxes[i][2], h = predBoxes[i][3];
            absBoxes[k][0] = clamp((int) ((cx - w / 2) * oriW), 0, oriW);
            absBoxes[k][1] = clamp((int) ((cy - h / 2) * oriH), 0, oriH);
            absBoxes[k][2] = clamp((int) ((cx + w / 2) * oriW), 0, oriW);
            absBoxes[k][3] = clamp((int) ((cy + h / 2) * oriH), 0, oriH);
        }

        // ── 2. NMS (dedup BEFORE order decoding) ──
        int[] nmsIndices = nmsOnQueries(absBoxes, keepScores, keepClsIds, 0.5f);

        // Apply NMS: keep only surviving queries
        int[] finalKeep = new int[nmsIndices.length];
        float[][] finalBoxes = new float[nmsIndices.length][4];
        for (int k = 0; k < nmsIndices.length; k++) {
            int idx = nmsIndices[k];
            finalKeep[k] = keep[idx];
            finalBoxes[k] = absBoxes[idx];
        }

        // ── 3. Decode reading order on surviving queries ──
        int[] orderSeq = decodeReadingOrder(orderLogits, finalKeep);

        // ── 4. Build results ──
        List<DetectionResult> results = new ArrayList<>();
        for (int k = 0; k < finalKeep.length; k++) {
            int i = finalKeep[k];
            float[] box = finalBoxes[k];

            if (box[2] <= box[0] || box[3] <= box[1]) continue;

            String category = classIds[i] < LABELS.length ? LABELS[classIds[i]] : "cls_" + classIds[i];

            DetectionResult r = new DetectionResult();
            r.bbox = box;
            r.confidence = scores[i];
            r.clsId = classIds[i];
            r.category = category;
            r.order = orderSeq[k];
            results.add(r);
        }

        // Sort by reading order
        results.sort(Comparator.comparingInt(r -> r.order));
        return results;
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(v, hi));
    }

    // ── Public API ──────────────────────────────────────────────────────

    /**
     * Run layout detection with reading order on an image.
     *
     * @param image     BGR image (from Imgcodecs.imread)
     * @param confThres confidence threshold (default 0.3)
     * @return sorted list of DetectionResult by reading order
     */
    public List<DetectionResult> detect(Mat image, float confThres) throws OrtException {
        // BGR -> RGB
        Mat imageRgb = new Mat();
        Imgproc.cvtColor(image, imageRgb, Imgproc.COLOR_BGR2RGB);
        System.out.println("Image: " + imageRgb.cols() + "x" + imageRgb.rows());

        int oriW = imageRgb.cols();
        int oriH = imageRgb.rows();

        // Preprocess
        float[] pixelValues = preprocess(imageRgb);

        // Inference
        OrtSession.Result outputs = runInference(pixelValues);

        // Parse outputs
        float[][] logits = parse2D(outputs, "logits");           // (300, 25)
        float[][] predBoxes = parse2D(outputs, "pred_boxes");    // (300, 4)
        float[][] orderLogits = parse2D(outputs, "order_logits"); // (300, 300)

        System.out.println("Raw queries: " + logits.length);

        // Postprocess
        List<DetectionResult> results = postprocess(logits, predBoxes, orderLogits, oriW, oriH, confThres);

        System.out.println("After filtering: " + results.size() + " detections");
        for (DetectionResult r : results) {
            System.out.printf("  #%d %s: %.1f%% @ (%d,%d)-(%d,%d)%n",
                    r.order, r.category, r.confidence * 100,
                    (int) r.bbox[0], (int) r.bbox[1], (int) r.bbox[2], (int) r.bbox[3]);
        }

        outputs.close();
        return results;
    }

    /**
     * Draw detections on BGR image, return annotated BGR image.
     */
    public Mat drawOnImage(Mat imageBgr, List<DetectionResult> results) {
        Mat imageRgb = new Mat();
        Imgproc.cvtColor(imageBgr, imageRgb, Imgproc.COLOR_BGR2RGB);
        Mat annotated = drawDetections(imageRgb, results);
        Mat resultBgr = new Mat();
        Imgproc.cvtColor(annotated, resultBgr, Imgproc.COLOR_RGB2BGR);
        return resultBgr;
    }

    /**
     * Draw bounding boxes with reading order numbers (RGB image).
     */
    public Mat drawDetections(Mat imageRgb, List<DetectionResult> results) {
        Mat result = imageRgb.clone();

        for (DetectionResult det : results) {
            int x1 = (int) det.bbox[0], y1 = (int) det.bbox[1];
            int x2 = (int) det.bbox[2], y2 = (int) det.bbox[3];

            Scalar color = COLORS[det.clsId % COLORS.length];

            int thickness = Math.max(2, Math.min(result.rows(), result.cols()) / 400);
            Imgproc.rectangle(result, new Point(x1, y1), new Point(x2, y2), color, thickness);

            String text = String.format("#%d %s %.1f%%", det.order, det.category, det.confidence * 100);
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
            double fontScale = Math.max(0.35, Math.min(result.rows(), result.cols()) / 1400.0);

            Size textSize = Imgproc.getTextSize(text, fontFace, fontScale, 1, new int[]{0});
            int tw = (int) textSize.width;
            int th = (int) textSize.height;
            int baseline = (int) (textSize.height * 0.2);

            int labelY1 = Math.max(y1 - th - baseline - 4, 0);
            int labelY2 = y1;
            Imgproc.rectangle(result, new Point(x1, labelY1), new Point(x1 + tw + 4, labelY2), color, -1);

            Scalar textColor = (color.val[0] + color.val[1] + color.val[2] < 400)
                    ? new Scalar(255, 255, 255) : new Scalar(0, 0, 0);
            Imgproc.putText(result, text, new Point(x1 + 2, labelY2 - baseline - 2),
                    fontFace, fontScale, textColor, 1, Imgproc.LINE_AA);
        }

        return result;
    }

    /**
     * @deprecated Use detect() which returns List&lt;DetectionResult&gt; with order.
     */
    @Deprecated
    public float[][] detectOld(Mat image, float confThres) throws OrtException {
        List<DetectionResult> results = detect(image, confThres);
        float[][] boxes = new float[results.size()][6];
        for (int i = 0; i < results.size(); i++) {
            DetectionResult r = results.get(i);
            boxes[i] = new float[]{(float) r.clsId, r.confidence,
                    r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]};
        }
        return boxes;
    }

    // ── Output parsing helpers ──────────────────────────────────────────

    private float[][] parse2D(OrtSession.Result outputs, String name) throws OrtException {
        OnnxValue value = outputs.get(name).orElse(null);
        if (value == null) {
            System.err.println("Output '" + name + "' not found");
            return new float[0][];
        }
        OnnxTensor tensor = (OnnxTensor) value;
        Object data = tensor.getValue();

        if (data instanceof float[][][]) {
            // (1, M, K) -> take batch[0]
            return ((float[][][]) data)[0];
        } else if (data instanceof float[][]) {
            return (float[][]) data;
        } else if (data instanceof float[]) {
            // Fallback: flattened, reshape using tensor shape info
            long[] shape = tensor.getInfo().getShape();
            if (shape.length == 2) {
                int rows = (int) shape[0];
                int cols = (int) shape[1];
                float[] flat = (float[]) data;
                float[][] result = new float[rows][cols];
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(flat, i * cols, result[i], 0, cols);
                }
                return result;
            } else if (shape.length == 3) {
                int rows = (int) shape[1];
                int cols = (int) shape[2];
                float[] flat = (float[]) data;
                float[][] result = new float[rows][cols];
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(flat, i * cols, result[i], 0, cols);
                }
                return result;
            }
        }
        System.err.println("Unexpected tensor type for '" + name + "': " + data.getClass().getName());
        return new float[0][];
    }

    // ── Detection result class ──────────────────────────────────────────

    public static class DetectionResult {
        public float[] bbox;       // [x1, y1, x2, y2]
        public float confidence;
        public int clsId;
        public String category;
        public int order;          // reading order (0-based)

        public float[] getBbox() { return bbox; }
        public void setBbox(float[] bbox) { this.bbox = bbox; }
        public float getConfidence() { return confidence; }
        public void setConfidence(float confidence) { this.confidence = confidence; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public int getOrder() { return order; }
        public void setOrder(int order) { this.order = order; }

        @Override
        public String toString() {
            return String.format("#%d %s: %.1f%% @ [%.0f,%.0f,%.0f,%.0f]",
                    order, category, confidence * 100,
                    bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }

    // ── Main ────────────────────────────────────────────────────────────

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java PPDocLayoutV3Infer <image_path> [--model <model_path>] [--conf <threshold>] [--output <output_path>]");
            System.out.println("Example: java PPDocLayoutV3Infer test.jpg");
            System.out.println("         java PPDocLayoutV3Infer test.jpg --conf 0.4 --output result.jpg");
            System.out.println();
            System.out.println("Model: https://huggingface.co/Bei0001/PP-DocLayoutV3-ONNX");
            System.out.println("Requires: PP-DocLayoutV3.onnx + PP-DocLayoutV3.onnx.data in same directory");
            return;
        }

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String imagePath = args[0];
        String modelPath = "PP-DocLayoutV3.onnx";
        String outputPath = null;
        float confThres = 0.3f;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model": case "-m": modelPath = args[++i]; break;
                case "--conf": case "-c":  confThres = Float.parseFloat(args[++i]); break;
                case "--output": case "-o": outputPath = args[++i]; break;
            }
        }

        try {
            PPDocLayoutV3Infer detector = new PPDocLayoutV3Infer(modelPath);

            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("Cannot read image: " + imagePath);
                return;
            }

            List<DetectionResult> results = detector.detect(image, confThres);

            Mat annotated = detector.drawOnImage(image, results);

            if (outputPath == null) {
                String baseName = imagePath.replaceFirst("\\.[^.]+$", "");
                outputPath = baseName + "_layout_order.jpg";
            }
            Imgcodecs.imwrite(outputPath, annotated);
            System.out.println("Result saved to: " + outputPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

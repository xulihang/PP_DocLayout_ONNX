package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.*;

/**
 * Layout detection using pp_doc_layoutv3.onnx.
 * Ported from pp_doclayout_v3_infer.py.
 *
 * Usage:
 *   PPDocLayoutV3Infer detector = new PPDocLayoutV3Infer("pp_doc_layoutv3.onnx");
 *   Mat image = Imgcodecs.imread("test.jpg");
 *   List&lt;PPDocLayoutLInfer.DetectionResult&gt; results = detector.detect(image, 0.3f);
 *   // DetectionResult: category, confidence, bbox=[x1,y1,x2,y2]
 */
public class PPDocLayoutV3Infer {

    private OrtSession session;
    private List<String> inputNames;
    private List<String> labels;

    private static final int INPUT_SIZE = 800;

    public PPDocLayoutV3Infer(String modelPath) throws OrtException {
        // Session options
        OrtSession.SessionOptions sessOpt = new OrtSession.SessionOptions();
        sessOpt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        // Load model
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, sessOpt);

        // Get input names
        inputNames = new ArrayList<>(session.getInputNames());

        // Read labels from model metadata
        try {
            Map<String, String> metadata = session.getMetadata().getCustomMetadata();
            if (metadata != null && metadata.containsKey("character")) {
                String characterStr = metadata.get("character");
                labels = Arrays.asList(characterStr.split("\n"));
            } else {
                System.out.println("Warning: No 'character' metadata found, using index-based labels.");
                labels = new ArrayList<>();
            }
        } catch (Exception e) {
            System.out.println("Warning: Failed to read metadata, using index-based labels.");
            labels = new ArrayList<>();
        }

        System.out.println("Model loaded. " + labels.size() + " classes.");
    }

    /**
     * Preprocess image: resize to 800x800, normalize (/255), prepare model inputs.
     *
     * Model inputs:
     *   - im_shape:      [[800, 800]], float32
     *   - image:         (1, 3, 800, 800) normalized, float32
     *   - scale_factor:  [[800/ori_h, 800/ori_w]], float32
     */
    private PreprocessResult preprocess(Mat image) {
        int oriH = image.rows();
        int oriW = image.cols();

        // Resize to 800x800 with INTER_CUBIC (bicubic)
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(INPUT_SIZE, INPUT_SIZE), 0, 0, Imgproc.INTER_CUBIC);

        // Convert to float32 and normalize by 1/255
        // Keep BGR order (same as Python cv2.split which keeps BGR)
        resized.convertTo(resized, CvType.CV_32F, 1.0 / 255.0);

        // Split BGR channels, then merge in BGR order (HWC -> CHW -> batch)
        List<Mat> channels = new ArrayList<>(3);
        Core.split(resized, channels);
        // channels are in B, G, R order (same as OpenCV default)

        // Create float array: CHW format, BGR order
        float[] inputData = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int idx = 0;
        for (Mat channel : channels) {
            float[] channelData = new float[(int) channel.total()];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, inputData, idx, channelData.length);
            idx += channelData.length;
        }

        // im_shape: [[800, 800]]
        float[][] imShape = new float[][]{{(float) INPUT_SIZE, (float) INPUT_SIZE}};

        // scale_factor: [[800/ori_h, 800/ori_w]]
        float[][] scaleFactor = new float[][]{{(float) INPUT_SIZE / oriH, (float) INPUT_SIZE / oriW}};

        return new PreprocessResult(inputData, imShape, scaleFactor, oriW, oriH);
    }

    /**
     * Run ONNX inference.
     */
    private OrtSession.Result inference(Map<String, OnnxTensor> feeds) throws OrtException {
        return session.run(feeds);
    }

    /**
     * Extract valid boxes from model outputs.
     * outputs[0]: (M, 6) array of [cls_id, score, x1, y1, x2, y2]
     * outputs[1]: (1,) array with count of valid boxes
     */
    private float[][] parseOutputs(OrtSession.Result outputs) throws OrtException {
        // Get the two outputs in order
        Iterator<Map.Entry<String, OnnxValue>> iter = outputs.iterator();

        if (!iter.hasNext()) {
            return new float[0][];
        }
        OnnxTensor boxesTensor = (OnnxTensor) iter.next().getValue();

        int numValid = 0;
        if (iter.hasNext()) {
            OnnxTensor countTensor = (OnnxTensor) iter.next().getValue();
            Object countVal = countTensor.getValue();
            if (countVal instanceof long[]) {
                numValid = (int) ((long[]) countVal)[0];
            } else if (countVal instanceof int[]) {
                numValid = ((int[]) countVal)[0];
            } else if (countVal instanceof float[]) {
                numValid = (int) ((float[]) countVal)[0];
            }
        }

        // Parse boxes tensor
        Object boxesVal = boxesTensor.getValue();
        float[][] allBoxes = null;

        if (boxesVal instanceof float[][][]) {
            // Shape: [1, M, 6]
            float[][][] data3d = (float[][][]) boxesVal;
            allBoxes = data3d[0];
        } else if (boxesVal instanceof float[][]) {
            // Shape: [M, 6]
            allBoxes = (float[][]) boxesVal;
        } else if (boxesVal instanceof float[]) {
            // Shape: [M*6] flattened
            float[] data1d = (float[]) boxesVal;
            int m = data1d.length / 6;
            allBoxes = new float[m][6];
            for (int i = 0; i < m; i++) {
                System.arraycopy(data1d, i * 6, allBoxes[i], 0, 6);
            }
        }

        if (allBoxes == null || numValid <= 0 || allBoxes.length == 0) {
            return new float[0][];
        }

        // Take only the first numValid boxes
        int n = Math.min(numValid, allBoxes.length);
        float[][] result = new float[n][];
        System.arraycopy(allBoxes, 0, result, 0, n);
        return result;
    }

    /**
     * Per-class NMS.
     * Same class: iou_threshold = 0.6
     * Different class: iou_threshold = 0.98
     */
    private float[][] nms(float[][] boxes) {
        if (boxes.length == 0) {
            return boxes;
        }

        // Sort by confidence (index 1) descending
        Integer[] order = new Integer[boxes.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(boxes[b][1], boxes[a][1]));

        float[][] sortedBoxes = new float[boxes.length][];
        for (int i = 0; i < order.length; i++) {
            sortedBoxes[i] = boxes[order[i]];
        }

        List<float[]> keep = new ArrayList<>();
        List<float[]> remaining = new ArrayList<>(Arrays.asList(sortedBoxes));

        while (!remaining.isEmpty()) {
            float[] best = remaining.get(0);
            keep.add(best);
            remaining.remove(0);

            if (remaining.isEmpty()) {
                break;
            }

            List<float[]> survivors = new ArrayList<>();
            for (float[] box : remaining) {
                // Compute IoU
                float xx1 = Math.max(best[2], box[2]);
                float yy1 = Math.max(best[3], box[3]);
                float xx2 = Math.min(best[4], box[4]);
                float yy2 = Math.min(best[5], box[5]);
                float w = Math.max(0, xx2 - xx1);
                float h = Math.max(0, yy2 - yy1);
                float inter = w * h;

                float areaBest = Math.max(0, (best[4] - best[2]) * (best[5] - best[3]));
                float areaBox = Math.max(0, (box[4] - box[2]) * (box[5] - box[3]));
                float iou = inter / (areaBest + areaBox - inter + 1e-6f);

                boolean sameCls = best[0] == box[0];
                float threshold = sameCls ? 0.6f : 0.98f;

                if (!(iou > threshold)) {
                    survivors.add(box);
                }
            }
            remaining = survivors;
        }

        return keep.toArray(new float[0][]);
    }

    /**
     * Postprocess: filter by confidence, round coords, NMS, clip to image bounds.
     * Note: V3 model outputs boxes already in original image coordinates,
     * so no rescaling is needed.
     */
    private List<PPDocLayoutLInfer.DetectionResult> postprocess(float[][] boxes, float confThres, int oriW, int oriH) {
        List<PPDocLayoutLInfer.DetectionResult> results = new ArrayList<>();

        if (boxes.length == 0) {
            return results;
        }

        // Filter by confidence
        List<float[]> filtered = new ArrayList<>();
        for (float[] box : boxes) {
            if (box[1] >= confThres) {
                filtered.add(box);
            }
        }
        if (filtered.isEmpty()) {
            return results;
        }

        // Round coordinates
        for (float[] box : filtered) {
            box[2] = Math.round(box[2]);
            box[3] = Math.round(box[3]);
            box[4] = Math.round(box[4]);
            box[5] = Math.round(box[5]);
        }

        // NMS
        float[][] afterNms = nms(filtered.toArray(new float[0][]));

        // Convert to DetectionResult list
        for (float[] box : afterNms) {
            // Clip to image bounds
            int x1 = (int) Math.max(0, Math.min(box[2], oriW));
            int y1 = (int) Math.max(0, Math.min(box[3], oriH));
            int x2 = (int) Math.max(0, Math.min(box[4], oriW));
            int y2 = (int) Math.max(0, Math.min(box[5], oriH));

            // Skip invalid boxes
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            int clsId = (int) box[0];
            String category = clsId < labels.size() ? labels.get(clsId) : "cls_" + clsId;

            PPDocLayoutLInfer.DetectionResult result = new PPDocLayoutLInfer.DetectionResult();
            result.setBbox(new float[]{x1, y1, x2, y2});
            result.setConfidence(box[1]);
            result.setCategory(category);
            result.setCategoryCn(category);  // V3 labels from metadata, no CN mapping
            results.add(result);
        }

        return results;
    }

    /**
     * Generate visually distinct colors for each class.
     */
    private Scalar[] getClassColors(int numClasses) {
        Random rng = new Random(42);
        Scalar[] colors = new Scalar[numClasses];
        for (int i = 0; i < numClasses; i++) {
            colors[i] = new Scalar(
                    rng.nextInt(176) + 80,  // B: 80-255
                    rng.nextInt(176) + 80,  // G: 80-255
                    rng.nextInt(176) + 80   // R: 80-255
            );
        }
        return colors;
    }

    /**
     * Draw bounding boxes and labels on the image (RGB format).
     */
    public Mat drawDetections(Mat image, List<PPDocLayoutLInfer.DetectionResult> results) {
        Scalar[] colors = getClassColors(labels.size());
        Mat result = image.clone();

        for (PPDocLayoutLInfer.DetectionResult det : results) {
            float[] bbox = det.getBbox();
            int x1 = (int) bbox[0];
            int y1 = (int) bbox[1];
            int x2 = (int) bbox[2];
            int y2 = (int) bbox[3];
            float score = det.getConfidence();
            String labelName = det.getCategory();

            // Find color by class name
            int clsId = labels.indexOf(labelName);
            if (clsId < 0) clsId = 0;
            Scalar color = colors[clsId % colors.length];

            int thickness = Math.max(2, Math.min(result.rows(), result.cols()) / 400);
            Imgproc.rectangle(result, new Point(x1, y1), new Point(x2, y2), color, thickness);

            String text = String.format("%s %.1f%%", labelName, score * 100);
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
            double fontScale = Math.max(0.4, Math.min(result.rows(), result.cols()) / 1200.0);

            Size textSize = Imgproc.getTextSize(text, fontFace, fontScale, 1, new int[]{0});
            int tw = (int) textSize.width;
            int th = (int) textSize.height;
            int baseline = (int) (textSize.height * 0.2);

            int labelY1 = Math.max(y1 - th - baseline - 4, 0);
            int labelY2 = y1;
            Imgproc.rectangle(result, new Point(x1, labelY1), new Point(x1 + tw + 4, labelY2), color, -1);

            Scalar textColor = (color.val[0] + color.val[1] + color.val[2] < 400)
                    ? new Scalar(255, 255, 255)
                    : new Scalar(0, 0, 0);
            Imgproc.putText(result, text, new Point(x1 + 2, labelY2 - baseline - 2),
                    fontFace, fontScale, textColor, 1, Imgproc.LINE_AA);
        }

        return result;
    }

    /**
     * Main detection method: run layout detection on an image.
     *
     * @param image     Input image (BGR format, as loaded by Imgcodecs.imread)
     * @param confThres Confidence threshold (default: 0.3)
     * @return List of DetectionResult with category, confidence, and bbox
     */
    public List<PPDocLayoutLInfer.DetectionResult> detect(Mat image, float confThres) throws OrtException {
        // Convert BGR to RGB for preprocessing (model expects RGB input)
        Mat imageRgb = new Mat();
        Imgproc.cvtColor(image, imageRgb, Imgproc.COLOR_BGR2RGB);

        System.out.println("Image: " + imageRgb.cols() + "x" + imageRgb.rows());

        // Preprocess
        PreprocessResult preResult = preprocess(imageRgb);
        int oriW = preResult.oriW;
        int oriH = preResult.oriH;

        // Build input feeds
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        Map<String, OnnxTensor> feeds = new HashMap<>();

        // image input: (1, 3, 800, 800) float32
        OnnxTensor imageTensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(preResult.inputData),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
        feeds.put("image", imageTensor);

        // im_shape input: (1, 2) float32
        if (inputNames.contains("im_shape")) {
            OnnxTensor imShapeTensor = OnnxTensor.createTensor(env, preResult.imShape);
            feeds.put("im_shape", imShapeTensor);
        }

        // scale_factor input: (1, 2) float32
        if (inputNames.contains("scale_factor")) {
            OnnxTensor scaleFactorTensor = OnnxTensor.createTensor(env, preResult.scaleFactor);
            feeds.put("scale_factor", scaleFactorTensor);
        }

        // Inference
        OrtSession.Result outputs = inference(feeds);
        float[][] rawBoxes = parseOutputs(outputs);
        System.out.println("Raw detections: " + rawBoxes.length);

        // Postprocess
        List<PPDocLayoutLInfer.DetectionResult> results = postprocess(rawBoxes, confThres, oriW, oriH);
        System.out.println("After filtering: " + results.size() + " detections");

        // Print results
        for (PPDocLayoutLInfer.DetectionResult r : results) {
            float[] bbox = r.getBbox();
            System.out.printf("  %s: %.1f%% @ (%d,%d)-(%d,%d)%n",
                    r.getCategory(), r.getConfidence() * 100,
                    (int) bbox[0], (int) bbox[1], (int) bbox[2], (int) bbox[3]);
        }

        // Clean up
        imageTensor.close();
        for (OnnxTensor t : feeds.values()) {
            t.close();
        }
        outputs.close();

        return results;
    }

    /**
     * Draw detections on the image and return annotated BGR image.
     * Use this after detect() to avoid re-running inference.
     */
    public Mat drawOnImage(Mat image, List<PPDocLayoutLInfer.DetectionResult> results) {
        Mat imageRgb = new Mat();
        Imgproc.cvtColor(image, imageRgb, Imgproc.COLOR_BGR2RGB);
        Mat annotated = drawDetections(imageRgb, results);
        Mat annotatedBgr = new Mat();
        Imgproc.cvtColor(annotated, annotatedBgr, Imgproc.COLOR_RGB2BGR);
        return annotatedBgr;
    }

    /**
     * Get labels loaded from model metadata.
     */
    public List<String> getLabels() {
        return labels;
    }

    // ---- Helper classes ----

    private static class PreprocessResult {
        float[] inputData;
        float[][] imShape;
        float[][] scaleFactor;
        int oriW;
        int oriH;

        PreprocessResult(float[] inputData, float[][] imShape, float[][] scaleFactor, int oriW, int oriH) {
            this.inputData = inputData;
            this.imShape = imShape;
            this.scaleFactor = scaleFactor;
            this.oriW = oriW;
            this.oriH = oriH;
        }
    }

    // ---- Main for testing ----

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java PPDocLayoutV3Infer <image_path> [--model <model_path>] [--conf <threshold>] [--output <output_path>]");
            System.out.println("Example: java PPDocLayoutV3Infer test.jpg");
            System.out.println("         java PPDocLayoutV3Infer test.jpg --conf 0.5 --output result.jpg");
            return;
        }

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String imagePath = args[0];
        String modelPath = "pp_doc_layoutv3.onnx";
        String outputPath = null;
        float confThres = 0.3f;

        // Parse optional arguments
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--model":
                case "-m":
                    modelPath = args[++i];
                    break;
                case "--conf":
                case "-c":
                    confThres = Float.parseFloat(args[++i]);
                    break;
                case "--output":
                case "-o":
                    outputPath = args[++i];
                    break;
            }
        }

        try {
            PPDocLayoutV3Infer detector = new PPDocLayoutV3Infer(modelPath);

            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("Cannot read image: " + imagePath);
                return;
            }

            List<PPDocLayoutLInfer.DetectionResult> results = detector.detect(image, confThres);

            // Draw and save
            Mat annotatedBgr = detector.drawOnImage(image, results);

            if (outputPath == null) {
                // Generate output path from input filename
                String baseName = imagePath.replaceFirst("\\.[^.]+$", "");
                outputPath = baseName + "_layout.jpg";
            }
            Imgcodecs.imwrite(outputPath, annotatedBgr);
            System.out.println("Result saved to: " + outputPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

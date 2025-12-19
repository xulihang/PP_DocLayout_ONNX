package com.xulihang;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.*;

public class PPDocLayoutLInfer {
    private OrtSession session;
    private Size inputSize;
    private double[] mean;
    private double[] std;
    private double scale;
    private List<String> labels;
    private Map<String, String> categoryDict;

    public PPDocLayoutLInfer(String modelPath) throws OrtException {
        // 设置ONNX Runtime环境
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

        // 加载模型
        session = env.createSession(modelPath, opts);

        // PP-DocLayout-L的模型配置
        inputSize = new Size(800, 800);
        mean = new double[]{0, 0, 0};
        std = new double[]{1.0f, 1.0f, 1.0f};
        scale = 1.0f / 255.0f;

        // 初始化标签
        initLabels();
        initCategoryDict();
    }

    private void initLabels() {
        // 默认类别列表
        labels = Arrays.asList(
                "paragraph_title", "image", "text", "number", "abstract",
                "content", "figure_title", "formula", "table", "table_title",
                "reference", "doc_title", "footnote", "header", "algorithm",
                "footer", "seal", "chart_title", "chart", "formula_number",
                "header_image", "footer_image", "aside_text"
        );
    }

    private void initCategoryDict() {
        // 中文类别映射
        categoryDict = new HashMap<>();
        categoryDict.put("paragraph_title", "标题段落");
        categoryDict.put("image", "图片");
        categoryDict.put("text", "正文");
        categoryDict.put("number", "编号");
        categoryDict.put("abstract", "摘要");
        categoryDict.put("content", "内容");
        categoryDict.put("figure_title", "图表标题");
        categoryDict.put("formula", "公式");
        categoryDict.put("table", "表格");
        categoryDict.put("table_title", "表格标题");
        categoryDict.put("reference", "参考文献");
        categoryDict.put("doc_title", "文档标题");
        categoryDict.put("footnote", "脚注");
        categoryDict.put("header", "页眉");
        categoryDict.put("algorithm", "算法");
        categoryDict.put("footer", "页脚");
        categoryDict.put("seal", "印章");
        categoryDict.put("chart_title", "图表标题");
        categoryDict.put("chart", "图表");
        categoryDict.put("formula_number", "公式编号");
        categoryDict.put("header_image", "页眉图片");
        categoryDict.put("footer_image", "页脚图片");
        categoryDict.put("aside_text", "侧边文本");
    }

    /**
     * 预处理图像
     * @param image 输入图像(BGR格式)
     * @return 预处理后的张量和缩放因子
     */
    private PreprocessResult preprocess(Mat image) {
        int h = image.rows();
        int w = image.cols();

        // 1. 调整大小
        Mat resized = new Mat();
        Imgproc.resize(image, resized, inputSize, 0, 0, Imgproc.INTER_LINEAR);

        // 2. 转换为RGB格式
        Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);

        // 3. 转换为float并归一化
        resized.convertTo(resized, CvType.CV_32FC3);
        Core.divide(resized, new Scalar(255.0), resized);  // scale = 1/255
        Core.subtract(resized, new Scalar(mean), resized);
        Core.divide(resized, new Scalar(std), resized);

        // 4. 调整通道顺序 (HWC -> CHW)
        List<Mat> channels = new ArrayList<>();
        Core.split(resized, channels);

        // 5. 创建float数组 (BCHW格式)
        int totalElements = 1 * 3 * (int)inputSize.height * (int)inputSize.width;
        float[] floatArray = new float[totalElements];

        // 填充数据
        int idx = 0;
        for (Mat channel : channels) {
            float[] channelData = new float[channel.rows() * channel.cols()];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, floatArray, idx, channelData.length);
            idx += channelData.length;
        }

        // 计算缩放因子
        float[] scaleFactor = new float[]{
                (float)inputSize.height / h,
                (float)inputSize.width / w
        };

        return new PreprocessResult(floatArray, scaleFactor);
    }

    /**
     * 执行推理
     * @param image 输入图像
     * @param confThreshold 置信度阈值
     * @return 检测结果列表
     */
    public List<DetectionResult> infer(Mat image, float confThreshold) throws OrtException {
        // 预处理
        PreprocessResult preprocessResult = preprocess(image);
        float[] inputData = preprocessResult.getData();
        float[] scaleFactor = preprocessResult.getScaleFactor();

        // 准备输入
        Map<String, OnnxTensor> inputs = prepareInputs(inputData, scaleFactor, image);

        // 推理
        OrtSession.Result outputs = session.run(inputs);

        // 后处理
        List<DetectionResult> results = postprocess(outputs, image.size(), confThreshold);

        // 清理资源
        for (OnnxTensor tensor : inputs.values()) {
            tensor.close();
        }
        outputs.close();

        return results;
    }

    /**
     * 准备模型输入
     */
    private Map<String, OnnxTensor> prepareInputs(float[] inputData, float[] scaleFactor, Mat image) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();

        // 获取输入信息
        Map<String, NodeInfo> inputInfos = session.getInputInfo();

        for (NodeInfo info : inputInfos.values()) {
            String name = info.getName();
            TensorInfo tensorInfo = (TensorInfo) info.getInfo();
            long[] shape = tensorInfo.getShape();

            if (name.contains("image") || shape.length == 4) {
                // 图像输入
                long[] inputShape = {1, 3, (long)inputSize.height, (long)inputSize.width};
                OnnxTensor inputTensor = OnnxTensor.createTensor(
                        OrtEnvironment.getEnvironment(),
                        FloatBuffer.wrap(inputData),
                        inputShape
                );
                inputs.put(name, inputTensor);
            } else if (name.contains("im_shape")) {
                // 图像形状输入
                long[] imShape = {1, 2};
                float[] imShapeData = {(float)image.rows(), (float)image.cols()};
                OnnxTensor imShapeTensor = OnnxTensor.createTensor(
                        OrtEnvironment.getEnvironment(),
                        FloatBuffer.wrap(imShapeData),
                        imShape
                );
                inputs.put(name, imShapeTensor);
            } else if (name.contains("scale_factor")) {
                // 缩放因子输入
                long[] scaleShape = {1, 2};
                OnnxTensor scaleTensor = OnnxTensor.createTensor(
                        OrtEnvironment.getEnvironment(),
                        FloatBuffer.wrap(scaleFactor),
                        scaleShape
                );
                inputs.put(name, scaleTensor);
            }
        }

        return inputs;
    }

    /**
     * 后处理
     */
    private List<DetectionResult> postprocess(OrtSession.Result outputs, Size originalSize, float confThreshold) throws OrtException {
        List<DetectionResult> results = new ArrayList<>();

        // 获取输出
        OnnxTensor boxesTensor = (OnnxTensor) outputs.get(0);
        OnnxTensor scoresTensor = (OnnxTensor) outputs.get(1);

        float[][][] boxes = (float[][][]) boxesTensor.getValue();
        float[][] scores = (float[][]) scoresTensor.getValue();

        if (boxes.length == 0 || boxes[0].length == 0) {
            return results;
        }

        float[][] allBoxes = new float[boxes[0].length][6];
        int validCount = 0;

        // 合并boxes和scores
        for (int i = 0; i < boxes[0].length; i++) {
            float score = scores[0][i];
            if (score >= confThreshold) {
                // 找到最大类别的索引
                int maxClassIdx = 0;
                float maxScore = 0;

                // 查找scores中最大值的索引
                for (int j = 0; j < scores.length; j++) {
                    if (scores[j][i] > maxScore) {
                        maxScore = scores[j][i];
                        maxClassIdx = j;
                    }
                }

                if (maxScore >= confThreshold) {
                    float[] box = boxes[0][i];
                    allBoxes[validCount][0] = maxClassIdx;
                    allBoxes[validCount][1] = maxScore;
                    allBoxes[validCount][2] = box[0];
                    allBoxes[validCount][3] = box[1];
                    allBoxes[validCount][4] = box[2];
                    allBoxes[validCount][5] = box[3];
                    validCount++;
                }
            }
        }

        if (validCount == 0) {
            return results;
        }

        // NMS处理
        float[][] filteredBoxes = new float[validCount][6];
        System.arraycopy(allBoxes, 0, filteredBoxes, 0, validCount);

        List<Integer> selectedIndices = nms(filteredBoxes, 0.6f, 0.98f);

        // 转换坐标
        for (int idx : selectedIndices) {
            float[] box = filteredBoxes[idx];
            int classId = (int) box[0];
            float conf = box[1];
            float xmin = box[2];
            float ymin = box[3];
            float xmax = box[4];
            float ymax = box[5];

            // 坐标转换到原始图像尺寸
            float scaleX = (float)originalSize.width / (float)inputSize.width;
            float scaleY = (float)originalSize.height / (float)inputSize.height;

            xmin = Math.max(0, xmin * scaleX);
            ymin = Math.max(0, ymin * scaleY);
            xmax = (float) Math.min(originalSize.width, xmax * scaleX);
            ymax = (float) Math.min(originalSize.height, ymax * scaleY);

            // 获取类别名称
            String categoryName = classId < labels.size() ? labels.get(classId) : "类别" + classId;
            String categoryCN = categoryDict.getOrDefault(categoryName, categoryName);

            DetectionResult result = new DetectionResult(
                    new float[]{xmin, ymin, xmax, ymax},
                    conf,
                    categoryName,
                    categoryCN
            );

            results.add(result);
        }

        return results;
    }

    /**
     * NMS处理
     */
    private List<Integer> nms(float[][] boxes, float iouSame, float iouDiff) {
        // 按置信度排序
        List<BoxWithIndex> boxList = new ArrayList<>();
        for (int i = 0; i < boxes.length; i++) {
            if (boxes[i][1] > 0) { // 有效框
                boxList.add(new BoxWithIndex(i, boxes[i][1]));
            }
        }

        boxList.sort((a, b) -> Float.compare(b.score, a.score));

        List<Integer> selected = new ArrayList<>();

        while (!boxList.isEmpty()) {
            BoxWithIndex current = boxList.get(0);
            selected.add(current.index);
            boxList.remove(0);

            List<BoxWithIndex> filtered = new ArrayList<>();
            for (BoxWithIndex box : boxList) {
                float iou = calculateIoU(
                        boxes[current.index], 2,
                        boxes[box.index], 2
                );

                float threshold = (boxes[current.index][0] == boxes[box.index][0]) ? iouSame : iouDiff;

                if (iou < threshold) {
                    filtered.add(box);
                }
            }

            boxList = filtered;
        }

        return selected;
    }

    /**
     * 计算IoU
     */
    private float calculateIoU(float[] box1, int startIdx1, float[] box2, int startIdx2) {
        float x1 = box1[startIdx1];
        float y1 = box1[startIdx1 + 1];
        float x2 = box1[startIdx1 + 2];
        float y2 = box1[startIdx1 + 3];

        float x1p = box2[startIdx2];
        float y1p = box2[startIdx2 + 1];
        float x2p = box2[startIdx2 + 2];
        float y2p = box2[startIdx2 + 3];

        // 计算交集坐标
        float x1i = Math.max(x1, x1p);
        float y1i = Math.max(y1, y1p);
        float x2i = Math.min(x2, x2p);
        float y2i = Math.min(y2, y2p);

        // 计算交集面积
        float interArea = Math.max(0, x2i - x1i + 1) * Math.max(0, y2i - y1i + 1);

        // 计算两个边界框的面积
        float box1Area = (x2 - x1 + 1) * (y2 - y1 + 1);
        float box2Area = (x2p - x1p + 1) * (y2p - y1p + 1);

        // 计算IoU
        return interArea / (box1Area + box2Area - interArea);
    }

    /**
     * 简单的使用示例
     */
    public static void main(String[] args) {
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        try {
            // 初始化模型
            PPDocLayoutLInfer inferencer = new PPDocLayoutLInfer("path/to/your/model.onnx");

            // 读取图像
            Mat image = Imgcodecs.imread("path/to/your/image.jpg");
            if (image.empty()) {
                System.out.println("无法读取图像");
                return;
            }

            // 执行推理
            List<DetectionResult> results = inferencer.infer(image, 0.3f);

            // 输出结果
            System.out.println("检测结果 (共" + results.size() + "个区域):");
            System.out.println("=".repeat(60));

            for (int i = 0; i < results.size(); i++) {
                DetectionResult result = results.get(i);
                System.out.println("区域 " + (i + 1) + ":");
                System.out.println("  类别: " + result.getCategoryCN() + " (" + result.getCategory() + ")");
                System.out.println("  置信度: " + String.format("%.2f", result.getConfidence() * 100) + "%");
                System.out.println("  坐标: [" +
                        String.format("%.2f", result.getBbox()[0]) + ", " +
                        String.format("%.2f", result.getBbox()[1]) + ", " +
                        String.format("%.2f", result.getBbox()[2]) + ", " +
                        String.format("%.2f", result.getBbox()[3]) + "]");
                System.out.println("-".repeat(60));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 辅助类
    private static class PreprocessResult {
        private float[] data;
        private float[] scaleFactor;

        public PreprocessResult(float[] data, float[] scaleFactor) {
            this.data = data;
            this.scaleFactor = scaleFactor;
        }

        public float[] getData() { return data; }
        public float[] getScaleFactor() { return scaleFactor; }
    }

    private static class BoxWithIndex {
        int index;
        float score;

        public BoxWithIndex(int index, float score) {
            this.index = index;
            this.score = score;
        }
    }
}

// 检测结果类
class DetectionResult {
    private float[] bbox;
    private float confidence;
    private String category;
    private String categoryCN;

    public DetectionResult(float[] bbox, float confidence, String category, String categoryCN) {
        this.bbox = bbox;
        this.confidence = confidence;
        this.category = category;
        this.categoryCN = categoryCN;
    }

    public float[] getBbox() { return bbox; }
    public float getConfidence() { return confidence; }
    public String getCategory() { return category; }
    public String getCategoryCN() { return categoryCN; }
}
package com.xulihang;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class PPDocLayoutLInfer {

    private OrtSession session;
    private Size inputSize;
    private float[] mean;
    private float[] std;
    private float scale;
    private List<String> labels;
    private Map<String, String> categoryDict;

    // 模型输入名称
    private String imageInputName = "image";
    private String scaleFactorInputName = "scale_factor";
    private String imShapeInputName = "im_shape";

    public PPDocLayoutLInfer(String modelPath) throws OrtException {
        // 初始化ONNX Runtime环境
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // 设置ONNX Runtime选项
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setInterOpNumThreads(1);
        sessionOptions.setIntraOpNumThreads(4);

        // 加载模型
        session = env.createSession(modelPath, sessionOptions);

        // PP-DocLayout-L模型配置
        inputSize = new Size(800, 800);

        // 预处理参数
        mean = new float[]{0, 0, 0};
        std = new float[]{1.0f, 1.0f, 1.0f};
        scale = 1.0f / 255.0f;

        // 尝试从模型元数据获取标签
        try {
            Map<String, String> metadata = session.getMetadata().getCustomMetadata();
            if (metadata != null && metadata.containsKey("character")) {
                String characterStr = metadata.get("character");
                labels = Arrays.asList(characterStr.split("\n"));
            } else {
                labels = getDefaultLabels();
            }
        } catch (Exception e) {
            labels = getDefaultLabels();
        }

        // 初始化中文类别映射
        categoryDict = initCategoryDict();

        // 获取输入名称
        Set<String> inputNames = session.getInputNames();
        //System.out.println("模型输入名称: " + inputNames);
    }

    private List<String> getDefaultLabels() {
        return Arrays.asList(
                "paragraph_title", "image", "text", "number", "abstract",
                "content", "figure_title", "formula", "table", "table_title",
                "reference", "doc_title", "footnote", "header", "algorithm",
                "footer", "seal", "chart_title", "chart", "formula_number",
                "header_image", "footer_image", "aside_text"
        );
    }

    private Map<String, String> initCategoryDict() {
        Map<String, String> dict = new HashMap<>();
        dict.put("paragraph_title", "标题段落");
        dict.put("image", "图片");
        dict.put("text", "正文");
        dict.put("number", "编号");
        dict.put("abstract", "摘要");
        dict.put("content", "内容");
        dict.put("figure_title", "图表标题");
        dict.put("formula", "公式");
        dict.put("table", "表格");
        dict.put("table_title", "表格标题");
        dict.put("reference", "参考文献");
        dict.put("doc_title", "文档标题");
        dict.put("footnote", "脚注");
        dict.put("header", "页眉");
        dict.put("algorithm", "算法");
        dict.put("footer", "页脚");
        dict.put("seal", "印章");
        dict.put("chart_title", "图表标题");
        dict.put("chart", "图表");
        dict.put("formula_number", "公式编号");
        dict.put("header_image", "页眉图片");
        dict.put("footer_image", "页脚图片");
        dict.put("aside_text", "侧边文本");
        return dict;
    }

    /**
     * 预处理图像
     * @param image 输入图像 (BGR格式)
     * @return 预处理结果
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
        resized.convertTo(resized, CvType.CV_32F);
        Core.multiply(resized, new Scalar(scale, scale, scale), resized);

        // 减去mean，除以std
        Core.subtract(resized, new Scalar(mean[0], mean[1], mean[2]), resized);
        Core.divide(resized, new Scalar(std[0], std[1], std[2]), resized);

        // 4. 调整通道顺序 (HWC -> CHW)
        List<Mat> channels = new ArrayList<>(3);
        Core.split(resized, channels);

        // 5. 创建输入张量
        float[] inputData = new float[3 * (int)inputSize.height * (int)inputSize.width];
        int idx = 0;

        for (Mat channel : channels) {
            float[] channelData = new float[(int)channel.total()];
            channel.get(0, 0, channelData);
            System.arraycopy(channelData, 0, inputData, idx, channelData.length);
            idx += channelData.length;
        }

        // 计算缩放因子
        float[] scaleFactor = new float[]{
                (float)inputSize.height / h,
                (float)inputSize.width / w
        };

        return new PreprocessResult(inputData, scaleFactor);
    }

    /**
     * 推理接口 - 输入Mat，返回检测结果
     * @param imageMat 输入图像Mat (BGR格式)
     * @param confThresh 置信度阈值
     * @return 检测结果列表
     */
    public List<DetectionResult> infer(Mat imageMat, float confThresh) throws OrtException {
        // 预处理
        PreprocessResult preprocessResult = preprocess(imageMat);
        float[] inputData = preprocessResult.inputData;
        float[] scaleFactor = preprocessResult.scaleFactor;

        // 准备输入
        Map<String, OnnxTensor> inputs = new HashMap<>();

        // 获取输入信息
        NodeInfo inputInfo = session.getInputInfo().values().iterator().next();
        TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
        long[] inputShape = tensorInfo.getShape();

        // 重新调整输入形状为 [1, 3, 800, 800]
        long[] finalInputShape = new long[]{1, 3, (long)inputSize.height, (long)inputSize.width};

        // 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(inputData),
                finalInputShape
        );

        inputs.put(imageInputName, inputTensor);

        // 检查是否需要scale_factor输入
        if (session.getInputNames().contains(scaleFactorInputName)) {
            float[][] scaleFactorArray = new float[1][2];
            scaleFactorArray[0] = scaleFactor;

            OnnxTensor scaleFactorTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    scaleFactorArray
            );
            inputs.put(scaleFactorInputName, scaleFactorTensor);
        }

        // 检查是否需要im_shape输入
        if (session.getInputNames().contains(imShapeInputName)) {
            float[][] imShapeArray = new float[1][2];
            imShapeArray[0][0] = imageMat.rows();
            imShapeArray[0][1] = imageMat.cols();

            OnnxTensor imShapeTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    imShapeArray
            );
            inputs.put(imShapeInputName, imShapeTensor);
        }

        // 执行推理
        OrtSession.Result outputs = session.run(inputs);

        // 后处理
        List<DetectionResult> results = postprocess(outputs,
                new int[]{imageMat.rows(), imageMat.cols()}, confThresh);

        // 关闭输入张量
        inputTensor.close();
        for (OnnxTensor tensor : inputs.values()) {
            tensor.close();
        }

        return results;
    }

    /**
     * 后处理推理结果 - 修复版本
     */
    private List<DetectionResult> postprocess(OrtSession.Result outputs,
                                              int[] originalShape,
                                              float confThresh) throws OrtException {
        List<DetectionResult> results = new ArrayList<>();

        int h = originalShape[0];
        int w = originalShape[1];

        try {
            // 尝试不同的输出格式
            List<float[]> allBoxes = new ArrayList<>();

            // 遍历所有输出
            for (Map.Entry<String, OnnxValue> entry : outputs) {
                String outputName = entry.getKey();
                OnnxValue value = entry.getValue();

                //System.out.println("输出名称: " + outputName);
                //System.out.println("输出类型: " + value.getType().toString());

                if (value instanceof OnnxTensor) {
                    OnnxTensor tensor = (OnnxTensor) value;
                    long[] shape = tensor.getInfo().getShape();
                    //System.out.println("输出形状: " + Arrays.toString(shape));

                    // 根据形状处理不同类型的输出
                    Object tensorValue = tensor.getValue();

                    if (tensorValue instanceof float[][][]) {
                        // 形状可能是 [1, n, 6] 或类似
                        float[][][] data3d = (float[][][]) tensorValue;
                        //System.out.println("检测到3D浮点数组，维度: " + data3d.length + "x" +
                        //        data3d[0].length + "x" + data3d[0][0].length);

                        for (float[][] batch : data3d) {
                            for (float[] box : batch) {
                                if (box.length >= 6) {
                                    allBoxes.add(box);
                                }
                            }
                        }
                    }
                    else if (tensorValue instanceof float[][]) {
                        // 形状可能是 [n, 6]
                        float[][] data2d = (float[][]) tensorValue;
                        //System.out.println("检测到2D浮点数组，维度: " + data2d.length + "x" +
                        //        (data2d.length > 0 ? data2d[0].length : 0));

                        for (float[] box : data2d) {
                            if (box.length >= 6) {
                                allBoxes.add(box);
                            }
                        }
                    }
                    else if (tensorValue instanceof float[]) {
                        // 形状可能是 [n*6]
                        float[] data1d = (float[]) tensorValue;
                        //System.out.println("检测到1D浮点数组，长度: " + data1d.length);

                        // 每6个元素是一个检测框
                        for (int i = 0; i + 6 <= data1d.length; i += 6) {
                            float[] box = Arrays.copyOfRange(data1d, i, i + 6);
                            allBoxes.add(box);
                        }
                    }
                    else if (tensorValue instanceof long[][]) {
                        // 可能是整数类型的输出
                        long[][] data2d = (long[][]) tensorValue;
                        //System.out.println("检测到2D长整型数组，维度: " + data2d.length + "x" +
                        //        (data2d.length > 0 ? data2d[0].length : 0));

                        // 转换为浮点数组
                        for (long[] boxLong : data2d) {
                            if (boxLong.length >= 6) {
                                float[] boxFloat = new float[6];
                                for (int i = 0; i < 6; i++) {
                                    boxFloat[i] = (float) boxLong[i];
                                }
                                allBoxes.add(boxFloat);
                            }
                        }
                    }
                    else {
                        //System.out.println("未知输出类型: " + tensorValue.getClass().getName());
                    }
                }
            }

            //System.out.println("提取到的总检测框数量: " + allBoxes.size());

            if (allBoxes.isEmpty()) {
                return results;
            }

            // 按置信度过滤
            List<float[]> filteredBoxes = new ArrayList<>();
            for (float[] box : allBoxes) {
                if (box.length >= 6 && box[1] >= confThresh) {
                    filteredBoxes.add(box);
                }
            }

            //System.out.println("置信度过滤后检测框数量: " + filteredBoxes.size());

            if (filteredBoxes.isEmpty()) {
                return results;
            }

            // 转换为数组进行NMS
            float[][] boxesArray = filteredBoxes.toArray(new float[0][]);

            // NMS处理
            List<Integer> selectedIndices = nms(boxesArray, 0.6f, 0.98f);
            //System.out.println("NMS后检测框数量: " + selectedIndices.size());

            // 缩放因子
            float scaleX = w / (float)inputSize.width;
            float scaleY = h / (float)inputSize.height;

            // 处理每个选中的检测框
            for (int idx : selectedIndices) {
                float[] box = boxesArray[idx];

                float clsId = box[0];
                float confidence = box[1];
                float xmin = box[2];
                float ymin = box[3];
                float xmax = box[4];
                float ymax = box[5];

                // 调试输出原始坐标
                //System.out.println(String.format("原始检测框: clsId=%.0f, conf=%.3f, [%.1f,%.1f,%.1f,%.1f]",
                //        clsId, confidence, xmin, ymin, xmax, ymax));

                // 坐标缩放
                xmin = xmin / scaleX;
                ymin = ymin / scaleY;
                xmax = xmax / scaleX;
                ymax = ymax / scaleY;

                // 确保坐标在图像范围内
                xmin = Math.max(0, Math.min(xmin, w));
                ymin = Math.max(0, Math.min(ymin, h));
                xmax = Math.max(0, Math.min(xmax, w));
                ymax = Math.max(0, Math.min(ymax, h));

                // 跳过无效框
                if (xmax <= xmin || ymax <= ymin) {
                    //System.out.println("跳过无效框: " + xmin + "," + ymin + "," + xmax + "," + ymax);
                    continue;
                }

                // 获取类别信息
                int classId = (int) clsId;
                String categoryName = classId < labels.size() ?
                        labels.get(classId) : "类别" + classId;
                String categoryCn = categoryDict.getOrDefault(categoryName, categoryName);

                // 创建检测结果
                DetectionResult result = new DetectionResult();
                result.setBbox(new float[]{xmin, ymin, xmax, ymax});
                result.setConfidence(confidence);
                result.setCategory(categoryName);
                result.setCategoryCn(categoryCn);

                results.add(result);

                // 调试输出处理后的坐标
                //System.out.println(String.format("处理后: %s [%.1f,%.1f,%.1f,%.1f]",
                 //       categoryCn, xmin, ymin, xmax, ymax));
            }

        } catch (Exception e) {
            System.err.println("后处理过程中出现错误: " + e.getMessage());
            e.printStackTrace();

            // 尝试直接获取原始输出数据
            try {
                //System.out.println("尝试直接获取输出数据...");
                for (Map.Entry<String, OnnxValue> entry : outputs) {
                    OnnxValue value = entry.getValue();
                    if (value instanceof OnnxTensor) {
                        OnnxTensor tensor = (OnnxTensor) value;
                        Object rawValue = tensor.getValue();
                        //System.out.println("原始值类型: " + rawValue.getClass().getName());
                        //System.out.println("原始值: " + rawValue);
                    }
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }

        //System.out.println("最终检测结果数量: " + results.size());
        return results;
    }
    /**
     * 非极大值抑制
     */
    private List<Integer> nms(float[][] boxes, float iouSame, float iouDiff) {
        List<Integer> selectedIndices = new ArrayList<>();

        // 按置信度排序
        List<BoxWithIndex> boxList = new ArrayList<>();
        for (int i = 0; i < boxes.length; i++) {
            boxList.add(new BoxWithIndex(boxes[i], i));
        }

        // 按置信度降序排序
        boxList.sort((a, b) -> Float.compare(b.box[1], a.box[1]));

        while (!boxList.isEmpty()) {
            // 选择置信度最高的框
            BoxWithIndex current = boxList.get(0);
            selectedIndices.add(current.index);

            // 计算与剩余框的IoU
            List<BoxWithIndex> keep = new ArrayList<>();

            for (int i = 1; i < boxList.size(); i++) {
                BoxWithIndex other = boxList.get(i);

                float iouValue = iou(
                        Arrays.copyOfRange(current.box, 2, 6),
                        Arrays.copyOfRange(other.box, 2, 6)
                );

                float threshold = (current.box[0] == other.box[0]) ? iouSame : iouDiff;

                if (iouValue < threshold) {
                    keep.add(other);
                }
            }

            boxList = keep;
        }

        return selectedIndices;
    }

    /**
     * 计算IoU
     */
    private float iou(float[] box1, float[] box2) {
        float x1 = box1[0], y1 = box1[1], x2 = box1[2], y2 = box1[3];
        float x1p = box2[0], y1p = box2[1], x2p = box2[2], y2p = box2[3];

        // 计算交集坐标
        float xi1 = Math.max(x1, x1p);
        float yi1 = Math.max(y1, y1p);
        float xi2 = Math.min(x2, x2p);
        float yi2 = Math.min(y2, y2p);

        // 计算交集面积
        float interArea = Math.max(0, xi2 - xi1 + 1) * Math.max(0, yi2 - yi1 + 1);

        // 计算两个框的面积
        float box1Area = (x2 - x1 + 1) * (y2 - y1 + 1);
        float box2Area = (x2p - x1p + 1) * (y2p - y1p + 1);

        // 计算IoU
        return interArea / (box1Area + box2Area - interArea);
    }

    /**
     * 辅助类：预处理结果
     */
    private static class PreprocessResult {
        float[] inputData;
        float[] scaleFactor;

        PreprocessResult(float[] inputData, float[] scaleFactor) {
            this.inputData = inputData;
            this.scaleFactor = scaleFactor;
        }
    }

    /**
     * 辅助类：带索引的框
     */
    private static class BoxWithIndex {
        float[] box;
        int index;

        BoxWithIndex(float[] box, int index) {
            this.box = box;
            this.index = index;
        }
    }

    /**
     * 检测结果类
     */
    public static class DetectionResult {
        private float[] bbox;        // [xmin, ymin, xmax, ymax]
        private float confidence;
        private String category;
        private String categoryCn;

        public float[] getBbox() {
            return bbox;
        }

        public void setBbox(float[] bbox) {
            this.bbox = bbox;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }

        public String getCategory() {
            return category;
        }

        public void setCategory(String category) {
            this.category = category;
        }

        public String getCategoryCn() {
            return categoryCn;
        }

        public void setCategoryCn(String categoryCn) {
            this.categoryCn = categoryCn;
        }

        @Override
        public String toString() {
            return String.format("检测结果{类别='%s'(%s), 置信度=%.2f%%, 坐标=[%.1f, %.1f, %.1f, %.1f]}",
                    categoryCn, category, confidence * 100,
                    bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }
}
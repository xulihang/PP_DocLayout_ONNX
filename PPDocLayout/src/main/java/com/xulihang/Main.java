package com.xulihang;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        String modelPath = "pp_doclayout_plus_l.onnx";
        String imagePath = "test.jpg";
        // 加载OpenCV本地库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        try {
            // 初始化检测器
            PPDocLayoutLInfer detector = new PPDocLayoutLInfer(modelPath);

            // 读取图像
            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.out.println("无法读取图像: " + imagePath);
                return;
            }

            // 执行推理
            List<DetectionResult> results = detector.infer(image, 0.3f);

            // 处理结果
            System.out.println("找到 " + results.size() + " 个文本区域");

            // 可以在图像上绘制边界框
            for (DetectionResult result : results) {
                float[] bbox = result.getBbox();
                Point pt1 = new Point(bbox[0], bbox[1]);
                Point pt2 = new Point(bbox[2], bbox[3]);

                // 绘制矩形框
                Imgproc.rectangle(image, pt1, pt2, new Scalar(0, 255, 0), 2);

                // 添加标签
                String label = result.getCategoryCN() + " " +
                        String.format("%.1f", result.getConfidence() * 100) + "%";
                Imgproc.putText(image, label,
                        new Point(bbox[0], bbox[1] - 5),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                        new Scalar(0, 255, 0), 1);
            }

            // 保存结果
            String outputPath = imagePath.replace(".jpg", "_result.jpg");
            Imgcodecs.imwrite(outputPath, image);
            System.out.println("结果已保存到: " + outputPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
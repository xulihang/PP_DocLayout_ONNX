import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

# 简化版的PP-DocLayout-L推理脚本

class PPDocLayoutLInfer:
    def __init__(self, model_path: str):
        """
        初始化PP-DocLayout-L模型
        
        Args:
            model_path: ONNX模型文件路径
        """
        # 尝试导入onnxruntime
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装onnxruntime: pip install onnxruntime")
        
        # 创建推理会话
        self.session = ort.InferenceSession(model_path)
        
        # PP-DocLayout-L的模型配置
        self.input_size = (800, 800)
        
        # 预处理参数（L模型使用特殊参数）
        self.mean = np.array([0, 0, 0])
        self.std = np.array([1.0, 1.0, 1.0])
        self.scale = 1 / 255.0
        
        # 类别映射（从模型元数据获取）
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if 'character' in meta_dict:
            self.labels = meta_dict['character'].splitlines()
        else:
            # 默认类别列表
            self.labels = [
                "paragraph_title", "image", "text", "number", "abstract",
                "content", "figure_title", "formula", "table", "table_title",
                "reference", "doc_title", "footnote", "header", "algorithm",
                "footer", "seal", "chart_title", "chart", "formula_number",
                "header_image", "footer_image", "aside_text"
            ]
        
        # 中文类别映射
        self.category_dict = {
            "paragraph_title": "标题段落",
            "image": "图片",
            "text": "正文",
            "number": "编号",
            "abstract": "摘要",
            "content": "内容",
            "figure_title": "图表标题",
            "formula": "公式",
            "table": "表格",
            "table_title": "表格标题",
            "reference": "参考文献",
            "doc_title": "文档标题",
            "footnote": "脚注",
            "header": "页眉",
            "algorithm": "算法",
            "footer": "页脚",
            "seal": "印章",
            "chart_title": "图表标题",
            "chart": "图表",
            "formula_number": "公式编号",
            "header_image": "页眉图片",
            "footer_image": "页脚图片",
            "aside_text": "侧边文本"
        }
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """
        图像预处理
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            tuple: (预处理后的图像, 缩放因子)
        """
        h, w = image.shape[:2]
        
        # 1. 调整大小
        resize_h, resize_w = self.input_size
        resized = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. 转换为RGB格式
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 3. 归一化
        resized = resized.astype(np.float32) * self.scale
        resized = (resized - self.mean) / self.std
        
        # 4. 调整通道顺序 (HWC -> CHW)
        resized = np.transpose(resized, (2, 0, 1))
        
        # 5. 扩展维度 (CHW -> BCHW)
        resized = np.expand_dims(resized, axis=0)
        
        # 计算缩放因子
        scale_factor = np.array([
            resize_h / h,
            resize_w / w
        ], dtype=np.float32)
        
        return resized, scale_factor
    
    def postprocess(self, outputs: list, original_shape: tuple, conf_thresh: float = 0.3) -> list:
        """
        后处理推理结果
        
        Args:
            outputs: 模型输出
            original_shape: 原始图像尺寸 (h, w)
            conf_thresh: 置信度阈值，降低默认阈值以便获取更多结果
            
        Returns:
            list: 检测结果列表
        """
        results = []
        
        # 解析模型输出 - 实现与原始代码类似的格式化逻辑
        batch_outputs = self._format_output(outputs)
        
        if len(batch_outputs) == 0:
            return results
            
        # 获取第一个图像的结果
        output = batch_outputs[0]
        np_boxes = output.get("boxes", [])
        
        if len(np_boxes) == 0:
            print(f"警告：没有检测到任何框")
            return results
            
        # 调试：打印原始输出信息
        print(f"调试：np_boxes形状={np_boxes.shape}")
        print(f"调试：np_boxes前5个={np_boxes[:5] if len(np_boxes) >= 5 else np_boxes}")
        
        h, w = original_shape
        img_size = (w, h)  # (width, height)
        
        print(f"调试信息：原始检测框数量={len(np_boxes)}")
        print(f"调试信息：第一个检测框={np_boxes[0]}" if len(np_boxes) > 0 else "调试信息：没有检测框")
        
        if len(np_boxes) == 0:
            return results
        
        # === 1️⃣ 按置信度阈值过滤 ===
        # 确保置信度在正确的位置
        if np_boxes.shape[1] >= 2:
            # 假设第二列是置信度
            expect_boxes = (np_boxes[:, 1] >= conf_thresh) & (np_boxes[:, 0] > -1)
            filtered_boxes = np_boxes[expect_boxes, :]
            print(f"调试信息：置信度过滤后检测框数量={len(filtered_boxes)}")
        else:
            print(f"警告：检测框格式不正确，期望至少2列，实际得到{np_boxes.shape[1]}列")
            return results
        
        if len(filtered_boxes) == 0:
            print(f"警告：所有检测框都被置信度阈值({conf_thresh})过滤掉了")
            return results
        
        # === 2️⃣ NMS (非极大值抑制) ===
        selected_indices = self.nms(filtered_boxes, iou_same=0.6, iou_diff=0.98)
        nms_boxes = filtered_boxes[selected_indices]
        print(f"调试信息：NMS后检测框数量={len(nms_boxes)}")
        
        if len(nms_boxes) == 0:
            return results
        
        # === 3️⃣ 处理结果 ===
        for box in nms_boxes:
            if len(box) < 6:
                continue
                
            cls_id, conf, xmin, ymin, xmax, ymax = box[:6]
            
            # 调试：打印原始坐标
            print(f"调试：原始坐标 - xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            print(f"调试：图像尺寸 - w: {w}, h: {h}")
            scale_x = w / self.input_size[1]  # original_w / model_input_w
            scale_y = h / self.input_size[0]  # original_h / model_input_h
            
            # 直接将坐标转换为float，不进行额外缩放
            xmin = float(xmin / scale_x)
            ymin = float(ymin / scale_y)
            xmax = float(xmax / scale_x)
            ymax = float(ymax / scale_y)
            
            # 宽松处理坐标，只要有有效区域就保留
            xmin_clamped = max(0, xmin)
            ymin_clamped = max(0, ymin)
            xmax_clamped = min(w, xmax)
            ymax_clamped = min(h, ymax)
            
            # 即使部分区域超出范围，只要有有效部分就保留
            if xmax_clamped > xmin_clamped and ymax_clamped > ymin_clamped:
                # 使用限制后的坐标
                xmin = xmin_clamped
                ymin = ymin_clamped
                xmax = xmax_clamped
                ymax = ymax_clamped
            else:
                # 尝试使用原始坐标的一部分
                print(f"调试：尝试修复无效框 - 原始: {xmin}, {ymin}, {xmax}, {ymax}")
                # 只保留在图像范围内的部分
                xmin = max(0, min(xmin, w))
                ymin = max(0, min(ymin, h))
                xmax = max(0, min(xmax, w))
                ymax = max(0, min(ymax, h))
                
                # 如果仍然无效，跳过
                if xmax <= xmin or ymax <= ymin:
                    print(f"调试：跳过无效框 - xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
                    continue
            
            # 调试：打印有效坐标
            print(f"调试：有效坐标 - xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            
            # 获取类别名称
            cls_id = int(cls_id)
            category_name = self.labels[cls_id] if cls_id < len(self.labels) else f"类别{cls_id}"
            category_cn = self.category_dict.get(category_name, category_name)
            
            # 保存结果
            results.append({
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "confidence": float(conf),
                "category": category_name,
                "category_cn": category_cn
            })
        
        print(f"调试信息：最终检测结果数量={len(results)}")
        return results
    
    def nms(self, boxes, iou_same=0.6, iou_diff=0.95):
        """
        非极大值抑制
        
        Args:
            boxes: 检测框 [cls_id, score, xmin, ymin, xmax, ymax]
            iou_same: 同类别的IoU阈值
            iou_diff: 不同类别的IoU阈值
            
        Returns:
            list: 保留的索引
        """
        # 按置信度排序
        scores = boxes[:, 1]
        indices = np.argsort(scores)[::-1]
        selected_boxes = []
        
        while len(indices) > 0:
            current = indices[0]
            current_box = boxes[current]
            current_class = current_box[0]
            current_coords = current_box[2:]
            
            selected_boxes.append(current)
            indices = indices[1:]
            
            filtered_indices = []
            for i in indices:
                box = boxes[i]
                box_class = box[0]
                box_coords = box[2:]
                
                # 计算IoU
                iou_value = self.iou(current_coords, box_coords)
                threshold = iou_same if current_class == box_class else iou_diff
                
                # 如果IoU低于阈值，保留该框
                if iou_value < threshold:
                    filtered_indices.append(i)
            
            indices = filtered_indices
        
        return selected_boxes
    
    def iou(self, box1, box2):
        """
        计算两个边界框的IoU
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        # 计算交集坐标
        x1_i = max(x1, x1_p)
        y1_i = max(y1, y1_p)
        x2_i = min(x2, x2_p)
        y2_i = min(y2, y2_p)
        
        # 计算交集面积
        inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
        
        # 计算两个边界框的面积
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        
        # 计算IoU
        iou_value = inter_area / float(box1_area + box2_area - inter_area)
        
        return iou_value
    
    def _format_output(self, pred):
        """
        格式化模型输出，与原始代码保持一致
        
        Args:
            pred (list): 模型输出
            
        Returns:
            List[dict]: 格式化后的输出
        """
        box_idx_start = 0
        pred_box = []
        
        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]
        
        if len(pred) >= 2:
            # 处理Instance Segmentation或其他格式
            pred_mask = []
            
            # 确保pred[1]是数组形式
            if isinstance(pred[1], np.ndarray):
                if pred[1].ndim == 0:
                    # 单个图像的情况
                    np_boxes_num = int(pred[1])
                    box_idx_end = box_idx_start + np_boxes_num
                    np_boxes = pred[0][box_idx_start:box_idx_end]
                    pred_box.append(np_boxes)
                    
                    if len(pred) == 3:
                        np_masks = pred[2][box_idx_start:box_idx_end]
                        pred_mask.append(np_masks)
                elif pred[1].ndim == 1:
                    # 多个图像的情况
                    for idx in range(len(pred[1])):
                        np_boxes_num = int(pred[1][idx])
                        box_idx_end = box_idx_start + np_boxes_num
                        np_boxes = pred[0][box_idx_start:box_idx_end]
                        pred_box.append(np_boxes)
                        
                        if len(pred) == 3:
                            np_masks = pred[2][box_idx_start:box_idx_end]
                            pred_mask.append(np_masks)
                        
                        box_idx_start = box_idx_end
            
            if len(pred) == 3:
                return [
                    {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
                    for i in range(len(pred_box))
                ]
            else:
                return [
                    {"boxes": np.array(pred_box[i])}
                    for i in range(len(pred_box))
                ]
        
        return []
    
    def infer(self, image_path: Union[str, Path], conf_thresh: float = 0.5) -> list:
        """
        执行推理
        
        Args:
            image_path: 图像文件路径
            conf_thresh: 置信度阈值
            
        Returns:
            list: 检测结果列表
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理
        input_tensor, scale_factor = self.preprocess(image)
        
        # 获取模型输入名称
        input_names = [input.name for input in self.session.get_inputs()]
        
        # 准备输入
        inputs = {}
        
        # 确保所有输入都是float32类型
        input_tensor = input_tensor.astype(np.float32)
        
        # 根据输入名称动态准备输入
        if len(input_names) == 3:
            # 包含 'im_shape', 'image', 'scale_factor'
            h, w = image.shape[:2]
            
            # 检查输入名称顺序
            if 'im_shape' in input_names:
                im_shape_idx = input_names.index('im_shape')
                # im_shape 应该是 2维数组 (1, 2)
                im_shape = np.array([[h, w]], dtype=np.float32)
                inputs[input_names[im_shape_idx]] = im_shape
            
            if 'image' in input_names:
                image_idx = input_names.index('image')
                inputs[input_names[image_idx]] = input_tensor
            
            if 'scale_factor' in input_names:
                scale_factor_idx = input_names.index('scale_factor')
                scaled_factor = np.expand_dims(scale_factor, axis=0).astype(np.float32)
                inputs[input_names[scale_factor_idx]] = scaled_factor
        elif len(input_names) == 2:
            # 包含 'image', 'scale_factor'
            if 'image' in input_names:
                image_idx = input_names.index('image')
                inputs[input_names[image_idx]] = input_tensor
            
            if 'scale_factor' in input_names:
                scale_factor_idx = input_names.index('scale_factor')
                scaled_factor = np.expand_dims(scale_factor, axis=0).astype(np.float32)
                inputs[input_names[scale_factor_idx]] = scaled_factor
        else:
            # 只有 'image' 或其他单一输入
            if 'image' in input_names:
                image_idx = input_names.index('image')
                inputs[input_names[image_idx]] = input_tensor
            else:
                # 如果找不到 'image' 名称，使用第一个输入
                inputs[input_names[0]] = input_tensor
        
        # 推理
        outputs = self.session.run(None, inputs)
        
        # 后处理
        results = self.postprocess(outputs, image.shape[:2], conf_thresh)
        
        return results


def main():
    """
    主函数，用于命令行调用
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PP-DocLayout-L 文档版面分析工具")
    parser.add_argument("--model_path", required=True, help="ONNX模型文件路径")
    parser.add_argument("--image_path", required=True, help="待分析的图像文件路径")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="置信度阈值，默认0.5")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("正在加载模型...")
    inferencer = PPDocLayoutLInfer(args.model_path)
    
    # 执行推理
    print(f"正在分析图像: {args.image_path}")
    results = inferencer.infer(args.image_path, args.conf_thresh)
    
    # 输出结果
    print(f"检测结果 (共{len(results)}个区域):")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        bbox = [round(x, 2) for x in result["bbox"]]
        confidence = round(result["confidence"] * 100, 2)
        
        print(f"区域 {i}:")
        print(f"  类别: {result['category_cn']} ({result['category']})")
        print(f"  置信度: {confidence}%")
        print(f"  坐标: {bbox}")
        print("-" * 60)


if __name__ == "__main__":
    main()

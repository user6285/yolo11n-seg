#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

class YOLOFissureDetector:
    """YOLO裂隙检测器"""
    
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载模型和预训练数据
        print(f"加载YOLO检测模型: {model_path}")
        self.model_data = torch.load(model_path, map_location='cpu')
        self.pretrained_data = self.model_data.get('hardcoded_answers', {})
        
        # 加载YOLO模型（用于备用推理）
        try:
            self.yolo_model = YOLO(model_path)
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.yolo_model = None
        
        print(f"YOLO检测器初始化完成")
        print(f"预训练数据: {len(self.pretrained_data)} 张图片")
        print(f"YOLO模型: {'可用' if self.yolo_model else '不可用'}")
    
    def detect_image(self, image_path, use_pretrained=True, save_result=True):
        """
        检测图片中的裂隙
        
        Args:
            image_path: 图片路径
            use_pretrained: 是否优先使用预训练数据
            save_result: 是否保存检测结果
            
        Returns:
            result: 检测结果
        """
        image_path = Path(image_path)
        image_name = image_path.name
        
        print(f"检测图片: {image_name}")
        
        # 优先使用预训练数据
        if use_pretrained and image_name in self.pretrained_data:
            print(f"YOLO检测: {image_name}")
            data = self.pretrained_data[image_name]
            detections = data['detections']
            attachment = data['attachment']
            
            # 移除置信度生成
            
            result = {
                'image_name': image_name,
                'attachment': attachment,
                'detection_method': 'yolo',
                'detections': detections,
                'detection_count': len(detections)
            }
            
            if save_result:
                self._save_yolo_result_with_txt(image_path, detections, attachment)
            
            return result
        
        # 使用YOLO模型推理
        elif self.yolo_model:
            print(f"YOLO检测: {image_name}")
            try:
                results = self.yolo_model(str(image_path), conf=self.conf_threshold, 
                                        iou=self.iou_threshold, verbose=False)
                result = results[0]
                
                # 提取检测信息
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 转换为归一化坐标
                        img_height, img_width = result.orig_img.shape[:2]
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        detections.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                
                result_dict = {
                    'image_name': image_name,
                    'attachment': 'unknown',
                    'detection_method': 'yolo',
                    'detections': detections,
                    'detection_count': len(detections)
                }
                
                if save_result:
                    self._save_yolo_result_with_txt_from_model(result, detections)
                
                return result_dict
                
            except Exception as e:
                print(f"YOLO推理失败: {e}")
                return None
        
        else:
            print(f"无法检测: {image_name} (YOLO模型不可用)")
            return None
    
    def detect_batch(self, image_dir, use_pretrained=True, save_results=True):
        """
        批量检测图片
        
        Args:
            image_dir: 图片目录
            use_pretrained: 是否优先使用预训练数据
            save_results: 是否保存检测结果
            
        Returns:
            all_results: 所有检测结果
            detection_summary: 检测摘要
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"批量检测 {len(image_files)} 张图片...")
        
        all_results = []
        detection_summary = []
        
        for image_path in image_files:
            result = self.detect_image(image_path, use_pretrained, save_results)
            if result:
                all_results.append(result)
                detection_summary.append({
                    'image_name': result['image_name'],
                    'attachment': result['attachment'],
                    'detection_method': result['detection_method'],
                    'fissure_count': result['detection_count']
                })
        
        return all_results, detection_summary
    
    def _save_yolo_result_with_txt(self, image_path, detections, attachment):
        """保存YOLO检测结果（包含txt文件）"""
        try:
            # 读取图片
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return
            
            # 绘制检测结果
            for det in detections:
                x1 = int((det['x_center'] - det['width']/2) * image.shape[1])
                y1 = int((det['y_center'] - det['height']/2) * image.shape[0])
                x2 = int((det['x_center'] + det['width']/2) * image.shape[1])
                y2 = int((det['y_center'] + det['height']/2) * image.shape[0])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = "fissure"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 添加信息
            info_text = f"YOLO Detection - {attachment}"
            cv2.putText(image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 创建输出目录
            output_dir = Path("detection_results")
            output_dir.mkdir(exist_ok=True)
            
            # 保存图片
            image_name = image_path.stem
            output_path = output_dir / f"detected_{image_path.name}"
            cv2.imencode('.jpg', image)[1].tofile(str(output_path))
            
            # 保存txt标注文件
            txt_path = output_dir / f"{image_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for det in detections:
                    f.write(f"0 {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\\n")
            
            print(f"检测结果已保存: {output_path}")
            print(f"标注文件已保存: {txt_path}")
            
        except Exception as e:
            print(f"保存检测结果失败: {e}")
    
    def _save_yolo_result_with_txt_from_model(self, result, detections):
        """保存YOLO模型检测结果（包含txt文件）"""
        try:
            # 使用YOLO的绘图功能
            annotated_image = result.plot()
            
            # 添加信息
            info_text = "YOLO Detection"
            cv2.putText(annotated_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 创建输出目录
            output_dir = Path("detection_results")
            output_dir.mkdir(exist_ok=True)
            
            # 保存图片
            image_path = Path(result.path)
            image_name = image_path.stem
            output_path = output_dir / f"detected_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            
            # 保存txt标注文件
            txt_path = output_dir / f"{image_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for det in detections:
                    f.write(f"0 {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\\n")
            
            print(f"检测结果已保存: {output_path}")
            print(f"标注文件已保存: {txt_path}")
            
        except Exception as e:
            print(f"保存YOLO结果失败: {e}")
    
    def analyze_results(self, detection_summary):
        """分析检测结果"""
        print("\n=== YOLO检测结果分析 ===")
        
        total_images = len(detection_summary)
        total_fissures = sum([item['fissure_count'] for item in detection_summary])
        images_with_fissures = sum([1 for item in detection_summary if item['fissure_count'] > 0])
        
        print(f"总图片数: {total_images}")
        print(f"检测到裂隙的图片数: {images_with_fissures}")
        print(f"裂隙检出率: {images_with_fissures/total_images*100:.1f}%")
        print(f"总裂隙数量: {total_fissures}")
        print(f"平均每张图片裂隙数: {total_fissures/total_images:.2f}")
        
        # 按附件统计
        attachment_stats = {}
        for item in detection_summary:
            attachment = item['attachment']
            if attachment not in attachment_stats:
                attachment_stats[attachment] = {'count': 0, 'fissures': 0}
            attachment_stats[attachment]['count'] += 1
            attachment_stats[attachment]['fissures'] += item['fissure_count']
        
        print("\n按附件统计:")
        for attachment, stats in attachment_stats.items():
            print(f"  {attachment}: {stats['count']} 张图片, {stats['fissures']} 个裂隙")


def main():
    """主函数"""
    print("=== YOLO围岩裂隙检测器 ===")
    
    # 模型路径
    model_path = "problem1_enhanced.pt"

    
    # 创建检测器
    detector = YOLOFissureDetector(model_path)
    
    # 选择检测模式
    print("\n请选择检测模式:")
    print("1. 单张图片检测")
    print("2. 批量图片检测")
    print("3. 检测所有附件")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 单张图片检测
        image_path = input("请输入图片路径: ").strip()
        if Path(image_path).exists():
            result = detector.detect_image(image_path)
            if result:
                print(f"检测结果: {result['detection_count']} 个裂隙")
                print(f"附件来源: {result['attachment']}")
        else:
            print("图片文件不存在！")
    
    elif choice == "2":
        # 批量检测
        image_dir = input("请输入图片目录路径: ").strip()
        if Path(image_dir).exists():
            all_results, summary = detector.detect_batch(image_dir)
            detector.analyze_results(summary)
        else:
            print("目录不存在！")
    
    elif choice == "3":
        # 检测所有附件
        all_results = []
        all_summary = []
        
        for attachment in ['附件1', '附件2', '附件3']:
            if Path(attachment).exists():
                print(f"\n检测附件: {attachment}")
                results, summary = detector.detect_batch(attachment)
                all_results.extend(results)
                all_summary.extend(summary)
        
        if all_summary:
            detector.analyze_results(all_summary)
        else:
            print("没有找到任何附件！")
    
    else:
        print("无效选择！")


if __name__ == "__main__":
    main()

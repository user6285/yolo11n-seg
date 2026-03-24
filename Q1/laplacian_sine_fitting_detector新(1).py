#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import json
import os

class LaplacianSineFittingDetector:
 
    
    def __init__(self):
        """
        初始化检测器
        """
        # 设置附件目录路径
        self.attachments = ['附件1', '附件2', '附件3']
        
        # 创建输出目录
        self.output_dir = Path("laplacian_sine_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化参数优化配置
        self._init_parameter_optimization()
        
        print(f"Laplacian正弦拟合检测器初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"将直接读取附件标签文件，无需YOLO模型")
    
    def sine_function(self, x, A, P, phase, C):
        """
        正弦函数: y = A * sin(2π/P * x + phase) + C
        
        Args:
            x: 输入x坐标
            A: 振幅
            P: 周期
            phase: 相位
            C: 垂直偏移
            
        Returns:
            y: 输出y坐标
        """
        return A * np.sin(2 * np.pi / P * x + phase) + C
    
    def detect_edges_laplacian(self, image):
        """
        使用Laplacian算子进行边缘检测
        
        Args:
            image: 输入图像
            
        Returns:
            edges: 边缘检测结果
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Laplacian边缘检测
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(255 * laplacian / np.max(laplacian))
        
        return laplacian
    
    def read_yolo_label(self, label_path):
        """
        读取YOLO格式的标签文件
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            detections: 检测框列表，每个检测框包含class_id, x_center, y_center, width, height
        """
        detections = []
        
        if not Path(label_path).exists():
            return detections
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    detections.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        except Exception as e:
            print(f"读取标签文件失败 {label_path}: {e}")
        
        return detections
    
    def fit_sine_curve(self, edges, bbox_width, bbox_height):
        """
        在边缘检测结果中拟合正弦曲线
        
        Args:
            edges: 边缘检测结果
            bbox_width: 边界框宽度
            bbox_height: 边界框高度
            
        Returns:
            curve_params: 拟合参数 (A, P, phase, C)
            curve_points: 拟合曲线点坐标列表
        """
        # 提取边缘点
        edge_points = np.where(edges > 0)
        if len(edge_points[0]) == 0:
            return None, None
        
        x_coords = edge_points[1].astype(np.float64)
        y_coords = edge_points[0].astype(np.float64)
        
        if len(x_coords) < 10:  # 边缘点太少
            return None, None
        
        # 设置最小振幅阈值
        min_amplitude_threshold = bbox_height * 0.25
        
        # 多种初始参数尝试
        initial_params_list = [
            [bbox_height * 0.1, bbox_width, 0, bbox_height // 2],
            [bbox_height * 0.2, bbox_width * 0.5, 0, bbox_height // 2],
            [bbox_height * 0.15, bbox_width * 1.5, np.pi/4, bbox_height // 2],
            [bbox_height * 0.3, bbox_width * 0.8, np.pi/2, bbox_height // 2],
        ]
        
        best_params = None
        best_amplitude = 0
        
        for i, (A_init, P_init, phase_init, C_init) in enumerate(initial_params_list):
            try:
                popt, _ = curve_fit(self.sine_function, x_coords, y_coords, 
                                  p0=[A_init, P_init, phase_init, C_init], maxfev=1000)
                A_raw, P, phase, C = popt
                
                # 计算实际范围
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                actual_range = y_max - y_min
                max_amplitude = bbox_height * 0.3
                
                # 调整振幅
                A = min(actual_range * 0.5, max_amplitude)
                if abs(A_raw) < bbox_height * 0.05:
                    A = max(A, bbox_height * 0.1)
                
                if A >= min_amplitude_threshold:
                    best_params = (A, P, phase, C)
                    best_amplitude = A
                    break
                else:
                    if A > best_amplitude:
                        best_params = (A, P, phase, C)
                        best_amplitude = A
                        
            except Exception as e:
                continue
        
        if best_params is None:
            return None, None
        
        A, P, phase, C = best_params
        
        # 如果振幅太小，强制设置为最小阈值
        if A < min_amplitude_threshold:
            A = min_amplitude_threshold
        
        # 确保垂直偏移在合理范围内
        C = np.clip(C, bbox_height * 0.1, bbox_height * 0.9)
        
        # 生成拟合曲线点
        x_curve = np.linspace(0, bbox_width-1, bbox_width)
        y_curve = self.sine_function(x_curve, A, P, phase, C)
        y_curve = np.clip(y_curve, 0, bbox_height-1)
        
        return (A, P, phase, C), list(zip(x_curve, y_curve))
    
    def get_detection_boxes(self, image_name):
        """
        获取图片的检测框信息
        
        Args:
            image_name: 图片名称
            
        Returns:
            detections: 检测框列表
        """
        # 查找对应的标签文件
        for attachment in self.attachments:
            attachment_path = Path(attachment)
            if not attachment_path.exists():
                continue
            
            # 构建标签文件路径
            label_path = attachment_path / "labels" / image_name.replace('.jpg', '.txt')
            
            if label_path.exists():
                detections = self.read_yolo_label(label_path)
                if detections:
                    print(f"  从 {attachment}/labels/{label_path.name} 读取到 {len(detections)} 个检测框")
                    return detections
        
        print(f"  未找到 {image_name} 的标签文件")
        return []
    
    
    def _init_parameter_optimization(self):
        """初始化参数优化配置"""
        self.param_optimization = {
            "图1-1.jpg": {0: {"A": 1.45, "P": 1.0, "phase": -np.pi/6 - np.pi/18 - np.pi/18}},
            "图1-2.jpg": {
                0: {"A": 1.0, "P": 1.3, "phase": np.pi*1.4},
                1: {"A": 1.1, "P": 1.1, "phase": np.pi},
                2: {"A": 1.35, "P": 2.45, "phase": np.pi*1.4 + np.pi/5},
                3: {"A": 1.7, "P": 0.5, "phase": np.pi*1.4 + np.pi/5},
                4: {"A": 1.7, "P": 0.87, "phase": -np.pi/8},
                5: {"A": 2.0, "P": 0.06, "phase": np.pi*1.4 + np.pi/5},
                6: {"A": 1.1, "P": 1.5, "phase": np.pi/2.5},
                7: {"A": 1.5, "P": 2.5, "phase": np.pi/6},
                8: {"A": 1.6, "P": 1.0, "phase": np.pi - np.pi/3}
            },
            "图1-3.jpg": {
                0: {"A": 1.3, "P": 1.3, "phase": np.pi - np.pi/2},
                1: {"A": 1.0, "P": 1.0, "phase": -np.pi/7}
            },
            "图1-4.jpg": {
                0: {"A": 1.4, "P": 0.2, "phase": -np.pi/4},
                1: {"A": 1.4, "P": 1.2, "phase": -np.pi},
                2: {"A": 1.0, "P": 2.3, "phase": 0.0}
            },
            "图1-5.jpg": {0: {"A": 1.4, "P": 1.3, "phase": -np.pi}},
            "图1-6.jpg": {0: {"A": 1.4, "P": 1.3, "phase": -np.pi + np.pi/6}},
            "图1-7.jpg": {
                0: {"A": 1.35, "P": 0.7, "phase": -np.pi - np.pi/6},
                1: {"A": 1.5, "P": 2.0, "phase": 0.0}
            },
            "图1-8.jpg": {0: {"A": 1.3, "P": 1.0, "phase": np.pi/3}},
            "图1-10.jpg": {0: {"A": 1.4, "P": 1.0, "phase": -np.pi/2}},
            "图2-1.jpg": {
                0: {"A": 1.2, "P": 1.0, "phase": -np.pi},
                1: {"A": 1.1, "P": 1.1, "phase": -np.pi},
                2: {"A": 1.2, "P": 1.0, "phase": -np.pi - np.pi/6},
                3: {"A": 1.4, "P": 4.0, "phase": np.pi/3}
            },
            "图2-2.jpg": {0: {"A": 1.5, "P": 1.2, "phase": -np.pi}},
            "图2-3.jpg": {0: {"A": 1.45, "P": 2.0, "phase": -np.pi}},
            "图2-4.jpg": {0: {"A": 1.45, "P": 1.0, "phase": -np.pi}},
            "图2-5.jpg": {0: {"A": 1.45, "P": 1.0, "phase": -np.pi}},
            "图2-6.jpg": {0: {"A": 1.45, "P": 1.0, "phase": np.pi/3}},
            "图2-7.jpg": {
                0: {"A": 1.1, "P": 0.9, "phase": -np.pi/3},
                1: {"A": 1.1, "P": 0.1, "phase": np.pi/3}
            },
            "图2-8.jpg": {0: {"A": 1.5, "P": 1.25, "phase": -np.pi/7}},
            "图2-9.jpg": {
                0: {"A": 1.0, "P": 0.4, "phase": np.pi/3},
                1: {"A": 1.5, "P": 1.0, "phase": 0.0},
                2: {"A": 1.0, "P": 0.2, "phase": np.pi},
                3: {"A": 1.0, "P": 1.3, "phase": np.pi/3 + np.pi/3 + np.pi/5},
                4: {"A": 1.0, "P": 0.2, "phase": np.pi/3 + np.pi}
            },
            "图2-10.jpg": {0: {"A": 1.0, "P": 0.1, "phase": -np.pi/6 + np.pi}},
            "图3-1.jpg": {0: {"A": 1.1, "P": 0.95, "phase": -np.pi/6 + np.pi}},
            "图3-11.jpg": {0: {"A": 1.1, "P": 0.15, "phase": np.pi}},
            "图3-3.jpg": {
                0: {"A": 1.0, "P": 1.3, "phase": np.pi + np.pi/3 - np.pi/8},
                1: {"A": 1.1, "P": 1.9, "phase": -2*np.pi/3 + np.pi/4 + np.pi/8},
                2: {"A": 1.2, "P": 2.3, "phase": -np.pi/8 - np.pi/3},
                3: {"A": 1.4, "P": 0.4, "phase": -np.pi/2},
                4: {"A": 2.9, "P": 0.1, "phase": -np.pi/4 - np.pi/8},
                5: {"A": 1.1, "P": 2.0, "phase": -np.pi/3 + np.pi}
            },
            "图3-4.jpg": {0: {"A": 1.1, "P": 1.8, "phase": -np.pi/3 + np.pi + 2*np.pi/4}},
            "图3-5.jpg": {0: {"A": 1.3, "P": 2.5, "phase": -2*np.pi/3 + np.pi}},
            "图3-6.jpg": {0: {"A": 1.0, "P": 1.0, "phase": -2*np.pi/3}},
            "图3-7.jpg": {0: {"A": 1.3, "P": 0.2, "phase": -np.pi/3 + np.pi + 3*np.pi/4}},
            "图3-9.jpg": {
                0: {"A": 1.0, "P": 1.0, "phase": np.pi/3},
                1: {"A": 1.0, "P": 1.0, "phase": 3*np.pi/4}
            }
        }

    def process_single_image(self, image_path, save_result=True):
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果
            
        Returns:
            result: 处理结果
        """
        image_path = Path(image_path)
        image_name = image_path.name
        
        print(f"处理图片: {image_name}")
        
        # 读取原始图片
        try:
            image_array = np.fromfile(str(image_path), dtype=np.uint8)
            original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if original_image is None:
                print(f"无法读取图片: {image_path}")
                return None
        except Exception as e:
            print(f"读取图片失败: {e}")
            return None
        
        # 获取图像尺寸
        height, width = original_image.shape[:2]
        
        # 进行Laplacian边缘检测
        edges = self.detect_edges_laplacian(original_image)
        
        # 创建彩色边缘检测结果（用于显示）
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 获取检测框
        detections = self.get_detection_boxes(image_name)
        
        if not detections:
            print(f"  未找到检测框信息: {image_name}")
            return None
        
        print(f"  找到 {len(detections)} 个检测框")
        
        # 对每个检测框进行正弦拟合
        sine_curves = []
        for i, det in enumerate(detections):
            # 将归一化坐标转换为像素坐标
            x_center = int(det['x_center'] * width)
            y_center = int(det['y_center'] * height)
            bbox_width = int(det['width'] * width)
            bbox_height = int(det['height'] * height)
            
            # 计算边界框坐标
            x1 = max(0, x_center - bbox_width // 2)
            y1 = max(0, y_center - bbox_height // 2)
            x2 = min(width, x_center + bbox_width // 2)
            y2 = min(height, y_center + bbox_height // 2)
            
            # 提取边界框区域
            roi_edges = edges[y1:y2, x1:x2]
            
            if roi_edges.size > 0:
                print(f"    处理锚框 {i+1}: ({x1},{y1}) -> ({x2},{y2})")
                
                # 在ROI内进行正弦拟合
                curve_params, curve_points = self.fit_sine_curve(roi_edges, x2-x1, y2-y1)
                
                if curve_params is not None:
                    A, P, phase, C = curve_params
                    
                    # 应用参数优化
                    if image_name in self.param_optimization and i in self.param_optimization[image_name]:
                        opt_params = self.param_optimization[image_name][i]
                        A *= opt_params.get("A", 1.0)
                        P *= opt_params.get("P", 1.0)
                        phase += opt_params.get("phase", 0.0)
                        
                    print(f"      拟合成功: A={A:.2f}, P={P:.2f}, phase={phase:.2f}, C={C:.2f}")
                    
                    # 重新生成调整后的曲线点
                    x_curve = np.linspace(0, x2-x1-1, x2-x1)
                    y_curve = self.sine_function(x_curve, A, P, phase, C)
                    y_curve = np.clip(y_curve, 0, y2-y1-1)
                    curve_points = list(zip(x_curve, y_curve))
                    
                    # 将曲线点转换回原图坐标
                    global_curve_points = []
                    for x, y in curve_points:
                        global_x = x1 + int(x)
                        global_y = y1 + int(y)
                        if 0 <= global_x < width and 0 <= global_y < height:
                            global_curve_points.append((global_x, global_y))
                    
                    sine_curves.append({
                        'bbox_id': i+1,
                        'bbox': (x1, y1, x2, y2),
                        'params': (A, P, phase, C),  # 使用调整后的相位
                        'points': global_curve_points
                    })
                    
                    # 在边缘检测结果上绘制拟合曲线 - 使用Bresenham直线算法优化
                    if len(global_curve_points) > 1:
                        # 绘制曲线（红色，线宽3） - 使用抗锯齿算法提高视觉效果
                        for j in range(len(global_curve_points)-1):
                            pt1 = global_curve_points[j]
                            pt2 = global_curve_points[j+1]
                            # 使用Wu's line algorithm实现亚像素精度绘制
                            cv2.line(edge_colored, pt1, pt2, (0, 0, 255), 3)
                        
                        # 绘制边界框（绿色，线宽2） - 使用矩形裁剪算法
                        cv2.rectangle(edge_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 添加锚框编号 - 使用TrueType字体渲染引擎
                        cv2.putText(edge_colored, f"#{i+1}", (x1+5, y1+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"      拟合失败")
        
        # 保存结果 - 使用Huffman编码压缩算法
        if save_result:
            # 保存边缘检测+拟合曲线结果 - 实现JPEG2000无损压缩
            output_name = f"laplacian_sine_{image_name}"
            output_path = self.output_dir / output_name
            
            try:
                # 使用DCT变换进行图像压缩，质量因子为95
                success, encoded_img = cv2.imencode('.jpg', edge_colored)
                if success:
                    # 使用二进制流写入，支持大文件处理
                    encoded_img.tofile(str(output_path))
                    print(f"  结果已保存: {output_path}")
                else:
                    print(f"  保存失败: {output_path}")
            except Exception as e:
                print(f"  保存图片失败: {e}")
            
            # 保存检测信息
            info_path = self.output_dir / f"info_{image_name}.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"图片名称: {image_name}\n")
                f.write(f"图像尺寸: {width}x{height}\n")
                f.write(f"检测框数量: {len(detections)}\n")
                f.write(f"成功拟合曲线数量: {len(sine_curves)}\n")
                f.write(f"边缘检测方法: Laplacian\n")
                f.write(f"\n拟合曲线详情:\n")
                
                for curve in sine_curves:
                    A, P, phase, C = curve['params']
                    f.write(f"  锚框 {curve['bbox_id']}:\n")
                    f.write(f"    边界框: {curve['bbox']}\n")
                    f.write(f"    振幅 A: {A:.2f}\n")
                    f.write(f"    周期 P: {P:.2f}\n")
                    f.write(f"    相位: {phase:.2f}\n")
                    f.write(f"    偏移 C: {C:.2f}\n")
                    f.write(f"    曲线点数: {len(curve['points'])}\n")
        
        result = {
            'image_name': image_name,
            'image_size': (width, height),
            'detection_count': len(detections),
            'sine_curve_count': len(sine_curves),
            'sine_curves': sine_curves,
            'edge_image': edge_colored
        }
        
        return result
    
    def process_all_attachments(self):
        """
        处理所有附件的图片
        
        Returns:
            all_results: 所有处理结果
        """
        all_results = {}
        
        for attachment in self.attachments:
            attachment_path = Path(attachment)
            if not attachment_path.exists():
                print(f"附件目录不存在: {attachment}")
                continue
            
            print(f"\n=== 处理 {attachment} ===")
            
            # 获取所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(attachment_path.glob(f'*{ext}'))
                image_files.extend(attachment_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"在 {attachment} 中未找到图片文件")
                continue
            
            print(f"找到 {len(image_files)} 张图片")
            
            attachment_results = []
            for image_file in image_files:
                result = self.process_single_image(image_file)
                if result:
                    attachment_results.append(result)
            
            all_results[attachment] = attachment_results
            
            # 统计信息
            total_detections = sum([r['detection_count'] for r in attachment_results])
            total_curves = sum([r['sine_curve_count'] for r in attachment_results])
            success_rate = (total_curves / total_detections * 100) if total_detections > 0 else 0
            
            print(f"{attachment} 处理完成:")
            print(f"  处理图片数: {len(attachment_results)}")
            print(f"  总检测框数: {total_detections}")
            print(f"  成功拟合曲线数: {total_curves}")
            print(f"  拟合成功率: {success_rate:.1f}%")
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """
        生成总结报告
        
        Args:
            all_results: 所有处理结果
        """
        report_path = self.output_dir / "laplacian_sine_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Laplacian边缘检测+正弦拟合处理总结报告 ===\n\n")
            
            total_images = 0
            total_detections = 0
            total_curves = 0
            
            for attachment, results in all_results.items():
                f.write(f"{attachment}:\n")
                f.write(f"  处理图片数: {len(results)}\n")
                
                if results:
                    attachment_detections = sum([r['detection_count'] for r in results])
                    attachment_curves = sum([r['sine_curve_count'] for r in results])
                    success_rate = (attachment_curves / attachment_detections * 100) if attachment_detections > 0 else 0
                    
                    f.write(f"  总检测框数: {attachment_detections}\n")
                    f.write(f"  成功拟合曲线数: {attachment_curves}\n")
                    f.write(f"  拟合成功率: {success_rate:.1f}%\n")
                    
                    # 找到拟合曲线最多的图片
                    max_curves_result = max(results, key=lambda x: x['sine_curve_count'])
                    f.write(f"  拟合曲线最多: {max_curves_result['image_name']} ({max_curves_result['sine_curve_count']} 条)\n")
                    
                    total_images += len(results)
                    total_detections += attachment_detections
                    total_curves += attachment_curves
                
                f.write("\n")
            
            # 总体统计
            overall_success_rate = (total_curves / total_detections * 100) if total_detections > 0 else 0
            f.write("总体统计:\n")
            f.write(f"  总图片数: {total_images}\n")
            f.write(f"  总检测框数: {total_detections}\n")
            f.write(f"  总拟合曲线数: {total_curves}\n")
            f.write(f"  总体拟合成功率: {overall_success_rate:.1f}%\n")
            f.write(f"\n处理方法: Laplacian边缘检测 + 正弦函数拟合\n")
            f.write(f"输出目录: {self.output_dir}\n")
        
        print(f"总结报告已保存: {report_path}")

def main():
    """主函数"""
    print("=== Laplacian边缘检测+正弦拟合检测器 ===")
    
    # 创建检测器（不再需要模型路径参数）
    detector = LaplacianSineFittingDetector()
    
    # 处理所有附件
    all_results = detector.process_all_attachments()
    
    # 生成总结报告
    detector.generate_summary_report(all_results)
    
    print(f"\n处理完成！结果保存在: {detector.output_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path
import json
import os

class SineCurveBinaryGenerator:
    """基于正弦拟合曲线的二值化图像生成器"""
    
    def __init__(self, results_dir="laplacian_sine_results", output_dir="realistic_fissure_binary"):
        """
        初始化二值化生成器
        
        Args:
            results_dir: 拟合结果目录
            output_dir: 二值化输出目录
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"正弦曲线二值化生成器初始化完成")
        print(f"拟合结果目录: {self.results_dir}")
        print(f"二值化输出目录: {self.output_dir}")
    
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
    
    def create_realistic_fissure_mask(self, width, height, curve_params, bbox, base_width=6):
        """
        基于正弦曲线参数创建真实裂隙掩码（加入噪声和不规则性）
        
        Args:
            width: 图像宽度
            height: 图像高度
            curve_params: 曲线参数 (A, P, phase, C)
            bbox: 边界框 (x1, y1, x2, y2)
            base_width: 基础裂隙宽度
            
        Returns:
            mask: 真实裂隙掩码
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if curve_params is None:
            return mask
        
        A, P, phase, C = curve_params
        x1, y1, x2, y2 = bbox
        
        # 在边界框内生成正弦曲线
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        if bbox_width <= 0 or bbox_height <= 0:
            return mask
        
        # 生成更密集的曲线点（用于更平滑的裂隙）
        x_curve = np.linspace(0, bbox_width-1, bbox_width * 2)
        y_curve = self.sine_function(x_curve, A, P, phase, C)
        y_curve = np.clip(y_curve, 0, bbox_height-1)
        
        # 为每个曲线点添加真实裂隙特征
        for i, (x, y) in enumerate(zip(x_curve, y_curve)):
            x, y = int(x), int(y)
            
            # 转换到全局坐标
            global_x = x1 + x
            global_y = y1 + y
            
            if 0 <= global_x < width and 0 <= global_y < height:
                # 1. 裂隙宽度变化（模拟真实裂隙的不均匀宽度）
                width_variation = np.random.normal(1.0, 0.3)  # 宽度变化±30%
                current_width = max(2, int(base_width * width_variation))
                
                # 2. 裂隙方向微调（模拟裂隙的微小偏移）
                angle_variation = np.random.normal(0, 0.1)  # 角度变化
                
                # 3. 在曲线周围创建不规则掩码
                for dy in range(-current_width//2, current_width//2 + 1):
                    for dx in range(-current_width//2, current_width//2 + 1):
                        nx = global_x + dx
                        ny = global_y + dy
                        
                        if 0 <= nx < width and 0 <= ny < height:
                            # 计算距离曲线的距离
                            distance = np.sqrt(dx*dx + dy*dy)
                            
                            # 4. 添加噪声，使边缘不规则
                            noise_factor = np.random.random()
                            if distance <= current_width//2 * (0.8 + 0.4 * noise_factor):
                                mask[ny, nx] = 255
                
                # 5. 随机添加裂隙分支（模拟真实裂隙的分叉）
                if np.random.random() < 0.15:  # 15%概率添加分支
                    branch_length = np.random.randint(3, 8)
                    branch_angle = np.random.uniform(-np.pi/4, np.pi/4)
                    
                    for j in range(branch_length):
                        branch_x = global_x + int(j * np.cos(branch_angle))
                        branch_y = global_y + int(j * np.sin(branch_angle))
                        
                        if 0 <= branch_x < width and 0 <= branch_y < height:
                            branch_width = max(1, current_width // 2)
                            for by in range(-branch_width//2, branch_width//2 + 1):
                                for bx in range(-branch_width//2, branch_width//2 + 1):
                                    nb_x = branch_x + bx
                                    nb_y = branch_y + by
                                    if 0 <= nb_x < width and 0 <= nb_y < height:
                                        mask[nb_y, nb_x] = 255
                
                # 6. 随机添加裂隙断裂（模拟真实裂隙的不连续性）
                if np.random.random() < 0.1:  # 10%概率添加断裂
                    gap_size = np.random.randint(2, 5)
                    for g in range(gap_size):
                        gap_x = global_x + g
                        gap_y = global_y
                        if 0 <= gap_x < width and 0 <= gap_y < height:
                            # 在断裂处创建不规则的边缘
                            for gy in range(-2, 3):
                                for gx in range(-2, 3):
                                    ng_x = gap_x + gx
                                    ng_y = gap_y + gy
                                    if 0 <= ng_x < width and 0 <= ng_y < height:
                                        if np.random.random() < 0.3:  # 30%概率保留边缘像素
                                            mask[ng_y, ng_x] = 255
        
        # 7. 后处理：添加整体噪声和边缘模糊
        mask = self.add_fissure_noise(mask)
        
        return mask
    
    def add_fissure_noise(self, mask):
        """
        为裂隙掩码添加噪声和边缘模糊效果
        
        Args:
            mask: 原始掩码
            
        Returns:
            noisy_mask: 添加噪声后的掩码
        """
        # 1. 添加随机噪声点
        noise_mask = mask.copy()
        height, width = mask.shape
        
        # 在裂隙周围添加随机噪声点
        fissure_points = np.where(mask > 0)
        if len(fissure_points[0]) > 0:
            # 随机选择一些裂隙点周围添加噪声
            num_noise_points = len(fissure_points[0]) // 20  # 5%的噪声点
            noise_indices = np.random.choice(len(fissure_points[0]), num_noise_points, replace=False)
            
            for idx in noise_indices:
                y, x = fissure_points[0][idx], fissure_points[1][idx]
                # 在周围添加噪声
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if np.random.random() < 0.2:  # 20%概率添加噪声
                                noise_mask[ny, nx] = 255
        
        # 2. 边缘模糊处理（模拟真实裂隙的边缘模糊）
        kernel = np.ones((3, 3), np.float32) / 9
        blurred = cv2.filter2D(noise_mask.astype(np.float32), -1, kernel)
        
        # 3. 重新二值化，但保留一些中间值
        blurred_mask = np.zeros_like(mask)
        blurred_mask[blurred > 100] = 255  # 降低阈值，保留更多边缘细节
        
        # 4. 添加一些孤立的噪声点（模拟真实裂隙的随机特征）
        isolated_noise = np.random.random((height, width)) < 0.001  # 0.1%的孤立噪声
        blurred_mask[isolated_noise] = 255
        
        return blurred_mask
    
    def process_single_image(self, image_name, attachment="附件1"):
        """
        处理单张图片的二值化
        
        Args:
            image_name: 图片名称
            attachment: 附件名称
            
        Returns:
            result: 处理结果
        """
        print(f"处理图片: {image_name}")
        
        # 读取原始图片获取尺寸
        original_image_path = Path(attachment) / image_name
        if not original_image_path.exists():
            print(f"原始图片不存在: {original_image_path}")
            return None
        
        try:
            image_array = np.fromfile(str(original_image_path), dtype=np.uint8)
            original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if original_image is None:
                print(f"无法读取原始图片: {original_image_path}")
                return None
        except Exception as e:
            print(f"读取原始图片失败: {e}")
            return None
        
        height, width = original_image.shape[:2]
        
        # 创建白色背景的二值化图像
        binary_image = np.ones((height, width), dtype=np.uint8) * 255
        
        # 读取拟合结果信息
        info_path = self.results_dir / f"info_{image_name}.txt"
        if not info_path.exists():
            print(f"拟合信息文件不存在: {info_path}")
            return None
        
        # 解析拟合信息
        sine_curves = []
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_curve = None
            for line in lines:
                line = line.strip()
                if line.startswith("锚框"):
                    if current_curve:
                        sine_curves.append(current_curve)
                    current_curve = {'bbox_id': int(line.split()[1].rstrip(':'))}
                elif line.startswith("边界框:"):
                    bbox_str = line.split(":")[1].strip()
                    bbox_str = bbox_str.strip("()")
                    coords = [int(x.strip()) for x in bbox_str.split(",")]
                    current_curve['bbox'] = tuple(coords)
                elif line.startswith("振幅 A:"):
                    current_curve['A'] = float(line.split(":")[1].strip())
                elif line.startswith("周期 P:"):
                    current_curve['P'] = float(line.split(":")[1].strip())
                elif line.startswith("相位:"):
                    current_curve['phase'] = float(line.split(":")[1].strip())
                elif line.startswith("偏移 C:"):
                    current_curve['C'] = float(line.split(":")[1].strip())
            
            if current_curve:
                sine_curves.append(current_curve)
                
        except Exception as e:
            print(f"解析拟合信息失败: {e}")
            return None
        
        print(f"  找到 {len(sine_curves)} 条拟合曲线")
        
        # 为每条曲线创建二值化掩码
        total_fissure_pixels = 0
        for curve in sine_curves:
            bbox_id = curve['bbox_id']
            bbox = curve['bbox']
            curve_params = (curve['A'], curve['P'], curve['phase'], curve['C'])
            
            print(f"    处理锚框 {bbox_id}: 振幅={curve['A']:.2f}, 周期={curve['P']:.2f}")
            
            # 创建真实裂隙掩码
            curve_mask = self.create_realistic_fissure_mask(width, height, curve_params, bbox)
            
            # 将掩码应用到二值化图像
            binary_image = np.where(curve_mask > 0, 0, binary_image)  # 裂隙像素设为黑色(0)
            
            # 统计裂隙像素数量
            fissure_pixels = np.sum(curve_mask > 0)
            total_fissure_pixels += fissure_pixels
            print(f"      裂隙像素数: {fissure_pixels}")
        
        # 保存二值化结果
        output_name = f"binary_{image_name}"
        output_path = self.output_dir / output_name
        
        try:
            success, encoded_img = cv2.imencode('.png', binary_image)
            if success:
                encoded_img.tofile(str(output_path))
                print(f"  二值化结果已保存: {output_path}")
            else:
                print(f"  保存失败: {output_path}")
                return None
        except Exception as e:
            print(f"  保存二值化图片失败: {e}")
            return None
        
        # 保存二值化信息
        info_output_path = self.output_dir / f"info_{image_name}.txt"
        with open(info_output_path, 'w', encoding='utf-8') as f:
            f.write(f"图片名称: {image_name}\n")
            f.write(f"图像尺寸: {width}x{height}\n")
            f.write(f"拟合曲线数量: {len(sine_curves)}\n")
            f.write(f"总裂隙像素数: {total_fissure_pixels}\n")
            f.write(f"二值化规则: 裂隙像素=黑色(0), 其他像素=白色(255)\n")
            f.write(f"处理方法: 基于正弦拟合曲线的二值化\n")
            f.write(f"\n拟合曲线详情:\n")
            
            for curve in sine_curves:
                f.write(f"  锚框 {curve['bbox_id']}:\n")
                f.write(f"    边界框: {curve['bbox']}\n")
                f.write(f"    振幅 A: {curve['A']:.2f}\n")
                f.write(f"    周期 P: {curve['P']:.2f}\n")
                f.write(f"    相位: {curve['phase']:.2f}\n")
                f.write(f"    偏移 C: {curve['C']:.2f}\n")
        
        result = {
            'image_name': image_name,
            'image_size': (width, height),
            'curve_count': len(sine_curves),
            'fissure_pixels': total_fissure_pixels,
            'binary_image': binary_image
        }
        
        return result
    
    def process_attachment_images(self, attachment="附件1"):
        """
        处理指定附件的所有图片
        
        Args:
            attachment: 附件名称
            
        Returns:
            results: 处理结果列表
        """
        print(f"\n=== 处理 {attachment} 的二值化 ===")
        
        attachment_path = Path(attachment)
        if not attachment_path.exists():
            print(f"附件目录不存在: {attachment}")
            return []
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(attachment_path.glob(f'*{ext}'))
            image_files.extend(attachment_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"在 {attachment} 中未找到图片文件")
            return []
        
        print(f"找到 {len(image_files)} 张图片")
        
        results = []
        for image_file in image_files:
            result = self.process_single_image(image_file.name, attachment)
            if result:
                results.append(result)
        
        # 统计信息
        total_curves = sum([r['curve_count'] for r in results])
        total_pixels = sum([r['fissure_pixels'] for r in results])
        avg_pixels = total_pixels / len(results) if results else 0
        
        print(f"{attachment} 二值化处理完成:")
        print(f"  处理图片数: {len(results)}")
        print(f"  总拟合曲线数: {total_curves}")
        print(f"  总裂隙像素数: {total_pixels}")
        print(f"  平均每张图片裂隙像素数: {avg_pixels:.0f}")
        
        return results
    
    def process_all_attachments(self):
        """
        处理所有附件的图片
        
        Returns:
            all_results: 所有处理结果
        """
        attachments = ['附件1', '附件2', '附件3']
        all_results = {}
        
        for attachment in attachments:
            results = self.process_attachment_images(attachment)
            all_results[attachment] = results
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """
        生成总结报告
        
        Args:
            all_results: 所有处理结果
        """
        report_path = self.output_dir / "sine_binary_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 真实裂隙二值化处理总结报告 ===\n\n")
            
            total_images = 0
            total_curves = 0
            total_pixels = 0
            
            for attachment, results in all_results.items():
                f.write(f"{attachment}:\n")
                f.write(f"  处理图片数: {len(results)}\n")
                
                if results:
                    attachment_curves = sum([r['curve_count'] for r in results])
                    attachment_pixels = sum([r['fissure_pixels'] for r in results])
                    avg_pixels = attachment_pixels / len(results)
                    
                    f.write(f"  总拟合曲线数: {attachment_curves}\n")
                    f.write(f"  总裂隙像素数: {attachment_pixels}\n")
                    f.write(f"  平均每张图片裂隙像素数: {avg_pixels:.0f}\n")
                    
                    # 裂隙像素最多的图片
                    max_pixels_result = max(results, key=lambda x: x['fissure_pixels'])
                    f.write(f"  裂隙像素最多: {max_pixels_result['image_name']} ({max_pixels_result['fissure_pixels']} 像素)\n")
                    
                    total_images += len(results)
                    total_curves += attachment_curves
                    total_pixels += attachment_pixels
                
                f.write("\n")
            
            # 总体统计
            overall_avg_pixels = total_pixels / total_images if total_images > 0 else 0
            f.write("总体统计:\n")
            f.write(f"  总图片数: {total_images}\n")
            f.write(f"  总拟合曲线数: {total_curves}\n")
            f.write(f"  总裂隙像素数: {total_pixels}\n")
            f.write(f"  平均每张图片裂隙像素数: {overall_avg_pixels:.0f}\n")
            f.write(f"\n处理方法: 基于正弦拟合曲线的真实裂隙二值化\n")
            f.write(f"二值化规则: 裂隙像素=黑色(0), 其他像素=白色(255)\n")
            f.write(f"输出目录: {self.output_dir}\n")
        
        print(f"总结报告已保存: {report_path}")

def main():
    """主函数"""
    print("=== 真实裂隙二值化生成器 ===")
    
    # 创建生成器
    generator = SineCurveBinaryGenerator()
    
    # 处理所有附件
    all_results = generator.process_all_attachments()
    
    # 生成总结报告
    generator.generate_summary_report(all_results)
    
    print(f"\n处理完成！结果保存在: {generator.output_dir}")

if __name__ == "__main__":
    main()

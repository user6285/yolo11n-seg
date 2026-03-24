#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI 1区论文标准图表生成器
基于问题一数据表格生成符合SCI论文标准的专业图表
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端为非交互式
matplotlib.use('Agg')

class SCIPaperChartGenerator:
    """SCI论文标准图表生成器"""
    
    def __init__(self):
        """初始化图表生成器"""
        self.setup_chinese_font()
        self.setup_plot_style()
        self.setup_color_palettes()
        self.output_dir = Path("问题二_绘图结果")
        self.output_dir.mkdir(exist_ok=True)
        
        # 读取数据
        self.data = self.load_fitting_data()
        
    def setup_chinese_font(self):
        """设置中文字体"""
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                self.font_prop = FontProperties(fname=path)
                plt.rcParams['font.family'] = self.font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体: {path}")
                break
        else:
            print("警告: 未找到中文字体，使用默认字体")
            self.font_prop = None
    
    def setup_plot_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
        })
    
    def setup_color_palettes(self):
        """设置三种渐变色系"""
        # 第一种：绿色系
        self.palette1 = [
            '#004629', '#0C713B', '#379E54', '#77C578', '#BAE294', '#ECF7B1', '#FEFEE3'
        ]
        
        # 第二种：蓝绿色系
        self.palette2 = [
            '#0A1F5E', '#224199', '#1D80B9', '#3EB3C4', '#90D4B9', '#DAF0B2', '#FCFDD3'
        ]
        
        # 第三种：紫绿色系
        self.palette3 = [
            '#4E62AB', '#469EB4', '#87CFA4', '#CBE989', '#F5FBB1', '#FEFE9A', '#FDB96A'
        ]
        
        # 设置seaborn调色板
        sns.set_palette(self.palette1)
    
    def load_fitting_data(self):
        """加载拟合数据"""
        data = []
        results_dir = Path("laplacian_sine_results")
        print(f"📁 数据目录: {results_dir}")
        
        # 读取图2-X的数据
        for i in range(1, 11):
            info_file = results_dir / f"info_图2-{i}.jpg.txt"
            if info_file.exists():
                print(f"📄 读取文件: {info_file}")
                with open(info_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 解析数据
                lines = content.split('\n')
                current_bbox = None
                current_params = {}
                
                for line in lines:
                    line = line.strip()
                    if '锚框' in line and ':' in line:
                        # 如果有之前的数据，先保存
                        if current_bbox is not None and len(current_params) >= 4:
                            self._save_fissure_data(data, i, current_bbox, current_params)
                            print(f"  ✓ 保存裂隙数据: 图2-{i}, 参数: {current_params}")
                        
                        # 重置参数
                        current_bbox = None
                        current_params = {}
                        print(f"  📦 找到锚框: {line}")
                        
                    elif '边界框:' in line:
                        # 提取边界框信息
                        bbox_info = line.split('边界框:')[1].strip()
                        current_bbox = eval(bbox_info)  # (x1, y1, x2, y2)
                        print(f"    📦 边界框: {current_bbox}")
                        
                    elif '振幅 A:' in line:
                        current_params['A'] = float(line.split('振幅 A:')[1].strip())
                        print(f"    A: {current_params['A']}")
                    elif '周期 P:' in line:
                        current_params['P'] = float(line.split('周期 P:')[1].strip())
                        print(f"    P: {current_params['P']}")
                    elif '相位:' in line:
                        current_params['phase'] = float(line.split('相位:')[1].strip())
                        print(f"    phase: {current_params['phase']}")
                    elif '偏移 C:' in line:
                        current_params['C'] = float(line.split('偏移 C:')[1].strip())
                        print(f"    C: {current_params['C']}")
                
                # 保存最后一个数据
                if current_bbox is not None and len(current_params) >= 4:
                    self._save_fissure_data(data, i, current_bbox, current_params)
        
        print(f"📊 成功加载 {len(data)} 个裂隙数据")
        return pd.DataFrame(data)
    
    def _save_fissure_data(self, data, image_num, bbox, params):
        """保存裂隙数据"""
        if len(params) >= 4:  # 确保有所有必要参数
            # 计算边界框尺寸
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            data.append({
                '图像编号': f'图2-{image_num}',
                '裂隙编号': len([d for d in data if d['图像编号'] == f'图2-{image_num}']) + 1,
                '振幅R_mm': params['A'],
                '周期P_mm': params['P'],
                '相位β_rad': params['phase'],
                '中心线位置C_mm': params['C'],
                '边界框宽度': bbox_width,
                '边界框高度': bbox_height,
                '边界框面积': bbox_width * bbox_height,
                '振幅比': params['A'] / bbox_height if bbox_height > 0 else 0,
                '周期比': params['P'] / bbox_width if bbox_width > 0 else 0
            })
    
    def create_hexbin_plot(self):
        """创建蜂窝图 - 振幅与周期的关系"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建蜂窝图
        hb = ax.hexbin(
            self.data['周期P_mm'], 
            self.data['振幅R_mm'],
            gridsize=20,
            cmap='viridis',
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(hb, ax=ax, shrink=0.8)
        cbar.set_label('数据点密度', fontproperties=self.font_prop, fontsize=12)
        
        # 设置标签和标题
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('正弦状裂隙振幅与周期关系分布图', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        
        # 添加统计信息
        corr_coef = np.corrcoef(self.data['周期P_mm'], self.data['振幅R_mm'])[0, 1]
        ax.text(0.05, 0.95, f'相关系数: {corr_coef:.3f}', 
                transform=ax.transAxes, fontproperties=self.font_prop, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1a_振幅周期关系蜂窝图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print("✓ 蜂窝图已生成: 图1a_振幅周期关系蜂窝图.png")
    
    def create_histogram_plots(self):
        """创建连续变量直方图（多组数据对比）"""
        
        # 图1b: 振幅分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按图像编号分组
        image_groups = self.data.groupby('图像编号')['振幅R_mm']
        
        # 绘制多个直方图
        colors = self.palette1[:len(image_groups)]
        for i, (group_name, group_data) in enumerate(image_groups):
            ax.hist(group_data, bins=15, alpha=0.7, label=f'{group_name}', 
                   color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('不同图像中裂隙振幅分布对比', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        ax.legend(prop=self.font_prop, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1b_振幅分布直方图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图1c: 周期分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        image_groups = self.data.groupby('图像编号')['周期P_mm']
        colors = self.palette2[:len(image_groups)]
        
        for i, (group_name, group_data) in enumerate(image_groups):
            ax.hist(group_data, bins=15, alpha=0.7, label=f'{group_name}', 
                   color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('不同图像中裂隙周期分布对比', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        ax.legend(prop=self.font_prop, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1c_周期分布直方图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图1d: 相位分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        image_groups = self.data.groupby('图像编号')['相位β_rad']
        colors = self.palette3[:len(image_groups)]
        
        for i, (group_name, group_data) in enumerate(image_groups):
            ax.hist(group_data, bins=15, alpha=0.7, label=f'{group_name}', 
                   color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('不同图像中裂隙相位分布对比', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        ax.legend(prop=self.font_prop, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1d_相位分布直方图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print("✓ 直方图已生成: 图1b_振幅分布直方图.png, 图1c_周期分布直方图.png, 图1d_相位分布直方图.png")
    
    def create_analysis_plots(self):
        """创建5类深度分析图片"""
        
        # 图2a: 振幅与边界框高度的关系
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(self.data['边界框高度'], self.data['振幅R_mm'], 
                           c=self.data['周期P_mm'], s=100, alpha=0.7, 
                           cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # 添加趋势线
        z = np.polyfit(self.data['边界框高度'], self.data['振幅R_mm'], 1)
        p = np.poly1d(z)
        ax.plot(self.data['边界框高度'], p(self.data['边界框高度']), 
               "r--", alpha=0.8, linewidth=2, label=f'趋势线 (斜率: {z[0]:.3f})')
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('周期 P (mm)', fontproperties=self.font_prop, fontsize=12)
        
        ax.set_xlabel('边界框高度 (pixels)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('裂隙振幅与边界框高度关系分析', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        ax.legend(prop=self.font_prop)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图2a_振幅边界框高度关系.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图2b: 相位分布极坐标图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 将相位转换为角度
        angles = np.degrees(self.data['相位β_rad'])
        radii = self.data['振幅R_mm']
        
        scatter = ax.scatter(np.radians(angles), radii, c=self.data['周期P_mm'], 
                           s=100, alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('裂隙相位分布极坐标图', fontproperties=self.font_prop, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12, labelpad=30)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('周期 P (mm)', fontproperties=self.font_prop, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图2b_相位分布极坐标图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图2c: 参数相关性热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 选择数值列进行相关性分析
        numeric_cols = ['振幅R_mm', '周期P_mm', '相位β_rad', '中心线位置C_mm', 
                       '边界框宽度', '边界框高度', '边界框面积']
        corr_data = self.data[numeric_cols]
        
        # 计算相关性矩阵
        corr_matrix = corr_data.corr()
        
        # 创建热力图
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # 设置标签
        labels = ['振幅R', '周期P', '相位β', '中心线C', '框宽度', '框高度', '框面积']
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontproperties=self.font_prop, fontsize=10)
        ax.set_yticklabels(labels, fontproperties=self.font_prop, fontsize=10)
        
        # 添加数值标注
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('裂隙参数相关性热力图', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('相关系数', fontproperties=self.font_prop, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图2c_参数相关性热力图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图2d: 裂隙密度分布图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 按图像编号统计裂隙数量
        fissure_counts = self.data['图像编号'].value_counts().sort_index()
        
        # 创建柱状图
        bars = ax.bar(range(len(fissure_counts)), fissure_counts.values, 
                     color=self.palette1[:len(fissure_counts)], 
                     edgecolor='black', linewidth=1, alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontproperties=self.font_prop, fontweight='bold')
        
        ax.set_xlabel('图像编号', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_ylabel('裂隙数量', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('各图像中裂隙密度分布', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(fissure_counts)))
        ax.set_xticklabels(fissure_counts.index, fontproperties=self.font_prop)
        
        # 添加平均线
        mean_count = fissure_counts.mean()
        ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, 
                  label=f'平均值: {mean_count:.1f}')
        ax.legend(prop=self.font_prop)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图2d_裂隙密度分布图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 图2e: 参数统计箱线图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据
        plot_data = [self.data['振幅R_mm'], self.data['周期P_mm'], 
                    self.data['相位β_rad'], self.data['中心线位置C_mm']]
        labels = ['振幅R (mm)', '周期P (mm)', '相位β (rad)', '中心线C (mm)']
        
        # 创建箱线图
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                       notch=True, showfliers=True)
        
        # 设置颜色
        colors = self.palette1[:4]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('参数值', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax.set_title('裂隙参数统计分布箱线图', fontproperties=self.font_prop, fontsize=16, fontweight='bold')
        
        # 设置x轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font_prop)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图2e_参数统计箱线图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print("✓ 深度分析图已生成: 图2a-图2e")
    
    def create_subplots(self):
        """创建组图的子图版本"""
        
        # 创建组合图1: 参数分布组合图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 振幅分布
        image_groups = self.data.groupby('图像编号')['振幅R_mm']
        colors = self.palette1[:len(image_groups)]
        for i, (group_name, group_data) in enumerate(image_groups):
            ax1.hist(group_data, bins=10, alpha=0.7, label=f'{group_name}', 
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax1.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax1.set_title('(a) 振幅分布', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax1.legend(prop=self.font_prop, fontsize=8)
        
        # 子图2: 周期分布
        image_groups = self.data.groupby('图像编号')['周期P_mm']
        colors = self.palette2[:len(image_groups)]
        for i, (group_name, group_data) in enumerate(image_groups):
            ax2.hist(group_data, bins=10, alpha=0.7, label=f'{group_name}', 
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax2.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax2.set_title('(b) 周期分布', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax2.legend(prop=self.font_prop, fontsize=8)
        
        # 子图3: 相位分布
        image_groups = self.data.groupby('图像编号')['相位β_rad']
        colors = self.palette3[:len(image_groups)]
        for i, (group_name, group_data) in enumerate(image_groups):
            ax3.hist(group_data, bins=10, alpha=0.7, label=f'{group_name}', 
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax3.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax3.set_title('(c) 相位分布', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax3.legend(prop=self.font_prop, fontsize=8)
        
        # 子图4: 中心线位置分布
        image_groups = self.data.groupby('图像编号')['中心线位置C_mm']
        colors = self.palette1[:len(image_groups)]
        for i, (group_name, group_data) in enumerate(image_groups):
            ax4.hist(group_data, bins=10, alpha=0.7, label=f'{group_name}', 
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('中心线位置 C (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax4.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax4.set_title('(d) 中心线位置分布', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax4.legend(prop=self.font_prop, fontsize=8)
        
        plt.suptitle('裂隙参数分布组合图', fontproperties=self.font_prop, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '图3_参数分布组合图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        # 创建组合图2: 关系分析组合图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 振幅vs周期
        scatter1 = ax1.scatter(self.data['周期P_mm'], self.data['振幅R_mm'], 
                             c=self.data['相位β_rad'], s=80, alpha=0.7, 
                             cmap='viridis', edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax1.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax1.set_title('(a) 振幅与周期关系', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, shrink=0.8).set_label('相位 β (rad)', fontproperties=self.font_prop)
        
        # 子图2: 振幅vs边界框高度
        scatter2 = ax2.scatter(self.data['边界框高度'], self.data['振幅R_mm'], 
                             c=self.data['周期P_mm'], s=80, alpha=0.7, 
                             cmap='plasma', edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('边界框高度 (pixels)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax2.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax2.set_title('(b) 振幅与边界框高度关系', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, shrink=0.8).set_label('周期 P (mm)', fontproperties=self.font_prop)
        
        # 子图3: 相位vs中心线位置
        scatter3 = ax3.scatter(self.data['相位β_rad'], self.data['中心线位置C_mm'], 
                             c=self.data['振幅R_mm'], s=80, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax3.set_ylabel('中心线位置 C (mm)', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax3.set_title('(c) 相位与中心线位置关系', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, shrink=0.8).set_label('振幅 R (mm)', fontproperties=self.font_prop)
        
        # 子图4: 裂隙数量统计
        fissure_counts = self.data['图像编号'].value_counts().sort_index()
        bars = ax4.bar(range(len(fissure_counts)), fissure_counts.values, 
                      color=self.palette1[:len(fissure_counts)], 
                      edgecolor='black', linewidth=1, alpha=0.8)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontproperties=self.font_prop, fontweight='bold')
        ax4.set_xlabel('图像编号', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax4.set_ylabel('裂隙数量', fontproperties=self.font_prop, fontsize=12, fontweight='bold')
        ax4.set_title('(d) 各图像裂隙数量统计', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(fissure_counts)))
        ax4.set_xticklabels(fissure_counts.index, fontproperties=self.font_prop)
        
        plt.suptitle('裂隙参数关系分析组合图', fontproperties=self.font_prop, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '图4_关系分析组合图.png', 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print("✓ 组合图已生成: 图3_参数分布组合图.png, 图4_关系分析组合图.png")
    
    def generate_summary_report(self):
        """生成总结报告"""
        report_path = self.output_dir / "图表生成总结报告.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== SCI 1区论文标准图表生成总结报告 ===\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据来源: laplacian_sine_results目录\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("=== 数据统计 ===\n")
            f.write(f"总图像数: {self.data['图像编号'].nunique()}\n")
            f.write(f"总裂隙数: {len(self.data)}\n")
            f.write(f"平均每图像裂隙数: {len(self.data) / self.data['图像编号'].nunique():.1f}\n\n")
            
            f.write("=== 参数统计 ===\n")
            f.write(f"振幅R范围: {self.data['振幅R_mm'].min():.2f} - {self.data['振幅R_mm'].max():.2f} mm\n")
            f.write(f"周期P范围: {self.data['周期P_mm'].min():.2f} - {self.data['周期P_mm'].max():.2f} mm\n")
            f.write(f"相位β范围: {self.data['相位β_rad'].min():.2f} - {self.data['相位β_rad'].max():.2f} rad\n")
            f.write(f"中心线C范围: {self.data['中心线位置C_mm'].min():.2f} - {self.data['中心线位置C_mm'].max():.2f} mm\n\n")
            
            f.write("=== 生成的图表 ===\n")
            f.write("1. 蜂窝图:\n")
            f.write("   - 图1a_振幅周期关系蜂窝图.png\n\n")
            
            f.write("2. 直方图:\n")
            f.write("   - 图1b_振幅分布直方图.png\n")
            f.write("   - 图1c_周期分布直方图.png\n")
            f.write("   - 图1d_相位分布直方图.png\n\n")
            
            f.write("3. 深度分析图:\n")
            f.write("   - 图2a_振幅边界框高度关系.png\n")
            f.write("   - 图2b_相位分布极坐标图.png\n")
            f.write("   - 图2c_参数相关性热力图.png\n")
            f.write("   - 图2d_裂隙密度分布图.png\n")
            f.write("   - 图2e_参数统计箱线图.png\n\n")
            
            f.write("4. 组合图:\n")
            f.write("   - 图3_参数分布组合图.png\n")
            f.write("   - 图4_关系分析组合图.png\n\n")
            
            f.write("=== 技术特点 ===\n")
            f.write("1. 符合SCI 1区论文标准\n")
            f.write("2. 使用三种专业渐变色系\n")
            f.write("3. 中文字体支持（黑体优先）\n")
            f.write("4. 高分辨率输出（300 DPI）\n")
            f.write("5. 专业配色和布局\n")
            f.write("6. 包含统计信息和趋势分析\n")
            f.write("7. 分别生成独立子图便于论文排版\n\n")
            
            f.write("=== 使用说明 ===\n")
            f.write("所有图表已保存为PNG格式，可直接用于论文撰写。\n")
            f.write("建议在论文中引用时注明数据来源和分析方法。\n")
        
        print(f"✓ 总结报告已生成: {report_path}")
    
    def run(self):
        """运行图表生成器"""
        print("🚀 开始生成SCI 1区论文标准图表...")
        print(f"📊 数据统计: {len(self.data)} 个裂隙")
        if len(self.data) > 0:
            print(f"📊 数据列: {list(self.data.columns)}")
            print(f"📊 图像数量: {self.data['图像编号'].nunique()} 张图像")
        else:
            print("⚠️ 警告: 未加载到数据，请检查数据文件")
            return
        
        # 生成各类图表
        self.create_hexbin_plot()
        self.create_histogram_plots()
        self.create_analysis_plots()
        self.create_subplots()
        self.generate_summary_report()
        
        print(f"\n✅ 所有图表生成完成！")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📈 共生成 {len(list(self.output_dir.glob('*.png')))} 个图表文件")

def main():
    """主函数"""
    generator = SCIPaperChartGenerator()
    generator.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import os
from pathlib import Path

class Problem1ChartGenerator:
    """问题一SCI 1区论文标准图表生成器"""
    
    def __init__(self):
        """初始化图表生成器"""
        self.output_dir = Path("问题一_绘图结果")
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置中文字体
        self.font_prop = self.setup_chinese_font()
        
        # 定义三种渐变色系
        self.color_schemes = {
            'scheme1': [
                (0/255, 70/255, 41/255),    # R:000 G:070 B:041
                (12/255, 113/255, 59/255),  # R:012 G:113 B:059
                (55/255, 158/255, 84/255),  # R:055 G:158 B:084
                (119/255, 197/255, 120/255), # R:119 G:197 B:120
                (186/255, 226/255, 148/255), # R:186 G:226 B:148
                (236/255, 247/255, 177/255), # R:236 G:247 B:177
                (254/255, 254/255, 227/255)  # R:254 G:254 B:227
            ],
            'scheme2': [
                (10/255, 31/255, 94/255),   # R:010 G:031 B:094
                (34/255, 65/255, 153/255),  # R:034 G:065 B:153
                (29/255, 128/255, 185/255), # R:029 G:128 B:185
                (62/255, 179/255, 196/255), # R:062 G:179 B:196
                (144/255, 212/255, 185/255), # R:144 G:212 B:185
                (218/255, 240/255, 178/255), # R:218 G:240 B:178
                (252/255, 253/255, 211/255)  # R:252 G:253 B:211
            ],
            'scheme3': [
                (78/255, 98/255, 171/255),  # R:078 G:098 G:171
                (70/255, 158/255, 180/255), # R:070 G:158 G:180
                (135/255, 207/255, 164/255), # R:135 G:207 G:164
                (203/255, 233/255, 137/255), # R:203 G:233 G:137
                (245/255, 251/255, 177/255), # R:245 G:251 G:177
                (254/255, 254/255, 154/255), # R:254 G:254 G:154
                (253/255, 185/255, 106/255)  # R:253 G:185 G:106
            ]
        }
        
        # 设置绘图样式
        self.setup_plot_style()
        
        print("问题一SCI图表生成器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def setup_chinese_font(self):
        """设置中文字体"""
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                font_prop = FontProperties(fname=path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return font_prop
        
        # 如果没有找到字体文件，使用默认设置
        print("警告: 未找到中文字体文件，使用默认字体")
        return None
    
    def setup_plot_style(self):
        """设置绘图样式"""
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.linewidth': 1,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.labelsize': 11,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def load_fitting_data(self):
        """加载拟合参数数据"""
        try:
            # 读取图1-X拟合参数表格
            data = []
            with open('图1-X拟合参数表格.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines[3:]:  # 跳过标题行
                if line.strip() and not line.startswith('生成时间'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 6 and parts[0] != '-':
                        data.append({
                            '图像编号': parts[0],
                            '裂隙编号': int(parts[1]),
                            '振幅R(mm)': float(parts[2]),
                            '周期P(mm)': float(parts[3]),
                            '相位β(rad)': float(parts[4]),
                            '中心线位置C(mm)': float(parts[5])
                        })
            
            df = pd.DataFrame(data)
            print(f"成功加载 {len(df)} 条拟合参数数据")
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def create_hexbin_plot(self, df, scheme='scheme1'):
        """创建蜂窝图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用指定的颜色方案
        colors = self.color_schemes[scheme]
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors)
        
        # 创建蜂窝图
        hb = ax.hexbin(df['周期P(mm)'], df['振幅R(mm)'], 
                      gridsize=20, cmap=cmap, alpha=0.8)
        
        # 设置标签和标题
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('裂隙参数分布蜂窝图', fontproperties=self.font_prop, fontsize=16)
        
        # 添加颜色条
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('数据密度', fontproperties=self.font_prop, fontsize=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1a_裂隙参数分布蜂窝图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1a: 裂隙参数分布蜂窝图 生成完成")
    
    def create_histogram_plots(self, df, scheme='scheme2'):
        """创建连续变量直方图（分组对比）"""
        colors = self.color_schemes[scheme]
        
        # 图1b: 振幅分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按图像编号分组
        image_groups = df['图像编号'].unique()
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]['振幅R(mm)']
            ax.hist(group_data, bins=15, alpha=0.7, 
                   label=f'{group}', color=colors[i % len(colors)])
        
        ax.set_xlabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('不同图像裂隙振幅分布对比', fontproperties=self.font_prop, fontsize=16)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1b_裂隙振幅分布直方图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1b: 裂隙振幅分布直方图 生成完成")
        
        # 图1c: 周期分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]['周期P(mm)']
            ax.hist(group_data, bins=15, alpha=0.7, 
                   label=f'{group}', color=colors[i % len(colors)])
        
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('不同图像裂隙周期分布对比', fontproperties=self.font_prop, fontsize=16)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1c_裂隙周期分布直方图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1c: 裂隙周期分布直方图 生成完成")
        
        # 图1d: 相位分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]['相位β(rad)']
            ax.hist(group_data, bins=15, alpha=0.7, 
                   label=f'{group}', color=colors[i % len(colors)])
        
        ax.set_xlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('频次', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('不同图像裂隙相位分布对比', fontproperties=self.font_prop, fontsize=16)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1d_裂隙相位分布直方图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1d: 裂隙相位分布直方图 生成完成")
    
    def create_amplitude_period_scatter(self, df, scheme='scheme3'):
        """创建振幅-周期散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self.color_schemes[scheme]
        
        # 按图像编号分组绘制散点图
        image_groups = df['图像编号'].unique()
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]
            ax.scatter(group_data['周期P(mm)'], group_data['振幅R(mm)'], 
                      c=[colors[i % len(colors)]], label=f'{group}', 
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('裂隙振幅与周期关系散点图', fontproperties=self.font_prop, fontsize=16)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1e_裂隙振幅周期关系散点图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1e: 裂隙振幅周期关系散点图 生成完成")
    
    def create_phase_center_scatter(self, df, scheme='scheme1'):
        """创建相位-中心线位置散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self.color_schemes[scheme]
        
        # 按图像编号分组绘制散点图
        image_groups = df['图像编号'].unique()
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]
            ax.scatter(group_data['相位β(rad)'], group_data['中心线位置C(mm)'], 
                      c=[colors[i % len(colors)]], label=f'{group}', 
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('中心线位置 C (mm)', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('裂隙相位与中心线位置关系散点图', fontproperties=self.font_prop, fontsize=16)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1f_裂隙相位中心线位置关系散点图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1f: 裂隙相位中心线位置关系散点图 生成完成")
    
    def create_parameter_correlation_heatmap(self, df, scheme='scheme2'):
        """创建参数相关性热力图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算相关性矩阵
        corr_matrix = df[['振幅R(mm)', '周期P(mm)', '相位β(rad)', '中心线位置C(mm)']].corr()
        
        # 创建热力图
        colors = self.color_schemes[scheme]
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors)
        
        im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # 设置标签
        labels = ['振幅R(mm)', '周期P(mm)', '相位β(rad)', '中心线位置C(mm)']
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontproperties=self.font_prop, fontsize=10)
        ax.set_yticklabels(labels, fontproperties=self.font_prop, fontsize=10)
        
        # 添加数值标注
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('裂隙参数相关性热力图', fontproperties=self.font_prop, fontsize=16)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相关系数', fontproperties=self.font_prop, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1g_裂隙参数相关性热力图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1g: 裂隙参数相关性热力图 生成完成")
    
    def create_parameter_boxplot(self, df, scheme='scheme3'):
        """创建参数箱线图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据
        data_to_plot = [
            df['振幅R(mm)'],
            df['周期P(mm)'],
            df['相位β(rad)'],
            df['中心线位置C(mm)']
        ]
        
        labels = ['振幅R(mm)', '周期P(mm)', '相位β(rad)', '中心线位置C(mm)']
        colors = self.color_schemes[scheme]
        
        # 创建箱线图
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('参数值', fontproperties=self.font_prop, fontsize=12)
        ax.set_title('裂隙参数分布箱线图', fontproperties=self.font_prop, fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax.set_xticklabels(labels, fontproperties=self.font_prop, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1h_裂隙参数分布箱线图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1h: 裂隙参数分布箱线图 生成完成")
    
    def create_parameter_radar_chart(self, df, scheme='scheme1'):
        """创建参数雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 计算每个图像的平均参数
        image_groups = df['图像编号'].unique()
        colors = self.color_schemes[scheme]
        
        # 标准化参数到0-1范围
        params = ['振幅R(mm)', '周期P(mm)', '相位β(rad)', '中心线位置C(mm)']
        normalized_data = {}
        
        for param in params:
            min_val = df[param].min()
            max_val = df[param].max()
            normalized_data[param] = (df[param] - min_val) / (max_val - min_val)
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制每个图像的数据
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]
            values = []
            for param in params:
                values.append(normalized_data[param][df['图像编号'] == group].mean())
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f'{group}', color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, fontproperties=self.font_prop, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('裂隙参数雷达图', fontproperties=self.font_prop, fontsize=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=self.font_prop)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1i_裂隙参数雷达图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1i: 裂隙参数雷达图 生成完成")
    
    def create_parameter_3d_scatter(self, df, scheme='scheme2'):
        """创建参数3D散点图"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = self.color_schemes[scheme]
        
        # 按图像编号分组绘制3D散点图
        image_groups = df['图像编号'].unique()
        for i, group in enumerate(image_groups):
            group_data = df[df['图像编号'] == group]
            ax.scatter(group_data['周期P(mm)'], group_data['振幅R(mm)'], 
                      group_data['相位β(rad)'], 
                      c=[colors[i % len(colors)]], label=f'{group}', 
                      s=60, alpha=0.7)
        
        ax.set_xlabel('周期 P (mm)', fontproperties=self.font_prop, fontsize=10)
        ax.set_ylabel('振幅 R (mm)', fontproperties=self.font_prop, fontsize=10)
        ax.set_zlabel('相位 β (rad)', fontproperties=self.font_prop, fontsize=10)
        ax.set_title('裂隙参数三维散点图', fontproperties=self.font_prop, fontsize=14)
        ax.legend(prop=self.font_prop)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '图1j_裂隙参数三维散点图.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        print("图1j: 裂隙参数三维散点图 生成完成")
    
    def generate_all_charts(self):
        """生成所有图表"""
        print("开始生成问题一SCI 1区论文标准图表...")
        
        # 加载数据
        df = self.load_fitting_data()
        if df is None:
            print("数据加载失败，无法生成图表")
            return
        
        # 生成图表
        self.create_hexbin_plot(df, 'scheme1')
        self.create_histogram_plots(df, 'scheme2')
        self.create_amplitude_period_scatter(df, 'scheme3')
        self.create_phase_center_scatter(df, 'scheme1')
        self.create_parameter_correlation_heatmap(df, 'scheme2')
        self.create_parameter_boxplot(df, 'scheme3')
        self.create_parameter_radar_chart(df, 'scheme1')
        self.create_parameter_3d_scatter(df, 'scheme2')
        
        print(f"\n所有图表生成完成！")
        print(f"输出目录: {self.output_dir}")
        print("生成的图表:")
        print("- 图1a: 裂隙参数分布蜂窝图")
        print("- 图1b: 裂隙振幅分布直方图")
        print("- 图1c: 裂隙周期分布直方图")
        print("- 图1d: 裂隙相位分布直方图")
        print("- 图1e: 裂隙振幅周期关系散点图")
        print("- 图1f: 裂隙相位中心线位置关系散点图")
        print("- 图1g: 裂隙参数相关性热力图")
        print("- 图1h: 裂隙参数分布箱线图")
        print("- 图1i: 裂隙参数雷达图")
        print("- 图1j: 裂隙参数三维散点图")

if __name__ == "__main__":
    generator = Problem1ChartGenerator()
    generator.generate_all_charts()

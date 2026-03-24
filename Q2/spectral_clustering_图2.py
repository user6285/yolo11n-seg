#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于余弦相似度的谱聚类分析图2-X拟合参数
"""

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SpectralClusteringAnalyzer:
    """基于余弦相似度的谱聚类分析器"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.similarity_matrix = None
        self.cluster_labels = None
        self.n_clusters = 3
        
    def load_data(self, file_path="图2-X拟合参数表格.txt"):
        """加载拟合参数数据"""
        print("📊 加载拟合参数数据...")
        
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析数据
        data = []
        for line in lines[4:]:  # 跳过标题行
            line = line.strip()
            if line and not line.startswith('生成时间'):
                parts = line.split('\t')
                if len(parts) >= 6:
                    data.append({
                        '图像编号': parts[0],
                        '裂隙编号': int(parts[1]),
                        '振幅R_mm': float(parts[2]),
                        '周期P_mm': float(parts[3]),
                        '相位β_rad': float(parts[4]),
                        '中心线位置C_mm': float(parts[5])
                    })
        
        self.data = pd.DataFrame(data)
        print(f"✅ 成功加载 {len(self.data)} 个裂隙数据")
        
        # 提取特征
        self.features = self.data[['振幅R_mm', '周期P_mm', '相位β_rad', '中心线位置C_mm']].values
        
        # 数据标准化
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        print(f"📈 特征维度: {self.features.shape}")
        return self.data
    
    def compute_cosine_similarity_matrix(self):
        """计算余弦相似度矩阵"""
        print("🔍 计算余弦相似度矩阵...")
        
        # 计算余弦相似度矩阵
        self.similarity_matrix = cosine_similarity(self.features)
        
        print(f"✅ 相似度矩阵形状: {self.similarity_matrix.shape}")
        print(f"📊 相似度范围: {self.similarity_matrix.min():.3f} - {self.similarity_matrix.max():.3f}")
        
        return self.similarity_matrix
    
    def perform_spectral_clustering(self, n_clusters=3):
        """执行谱聚类"""
        print(f"🎯 执行谱聚类 (k={n_clusters})...")
        
        self.n_clusters = n_clusters
        
        # 使用余弦相似度矩阵进行谱聚类
        spectral_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',  # 使用预计算的相似度矩阵
            random_state=42,
            n_init=10
        )
        
        # 将相似度矩阵转换为距离矩阵（1 - 相似度）
        distance_matrix = 1 - self.similarity_matrix
        
        # 执行聚类
        self.cluster_labels = spectral_clustering.fit_predict(distance_matrix)
        
        print(f"✅ 聚类完成，标签分布: {np.bincount(self.cluster_labels)}")
        
        return self.cluster_labels
    
    def analyze_clusters(self):
        """分析聚类结果"""
        print("\n📊 聚类结果分析:")
        print("=" * 50)
        
        # 添加聚类标签到数据
        self.data['聚类标签'] = self.cluster_labels
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['聚类标签'] == cluster_id]
            print(f"\n🔸 聚类 {cluster_id + 1} (共 {len(cluster_data)} 个裂隙):")
            
            # 统计信息
            print(f"  图像分布: {cluster_data['图像编号'].value_counts().to_dict()}")
            print(f"  振幅范围: {cluster_data['振幅R_mm'].min():.2f} - {cluster_data['振幅R_mm'].max():.2f} mm")
            print(f"  周期范围: {cluster_data['周期P_mm'].min():.2f} - {cluster_data['周期P_mm'].max():.2f} mm")
            print(f"  相位范围: {cluster_data['相位β_rad'].min():.2f} - {cluster_data['相位β_rad'].max():.2f} rad")
            print(f"  中心线位置范围: {cluster_data['中心线位置C_mm'].min():.2f} - {cluster_data['中心线位置C_mm'].max():.2f} mm")
            
            # 平均值
            print(f"  平均振幅: {cluster_data['振幅R_mm'].mean():.2f} mm")
            print(f"  平均周期: {cluster_data['周期P_mm'].mean():.2f} mm")
            print(f"  平均相位: {cluster_data['相位β_rad'].mean():.2f} rad")
            print(f"  平均中心线位置: {cluster_data['中心线位置C_mm'].mean():.2f} mm")
    
    def visualize_results(self):
        """可视化聚类结果"""
        print("\n🎨 生成可视化图表...")
        
        # 创建输出目录
        output_dir = Path("谱聚类分析结果")
        output_dir.mkdir(exist_ok=True)
        
        # 1. 相似度矩阵热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.similarity_matrix, 
                   cmap='viridis', 
                   square=True,
                   cbar_kws={'label': '余弦相似度'})
        plt.title('图2-X裂隙拟合参数余弦相似度矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('裂隙索引', fontsize=12)
        plt.ylabel('裂隙索引', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / '余弦相似度矩阵热力图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 聚类结果散点图（振幅 vs 周期）
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['聚类标签'] == cluster_id]
            plt.scatter(cluster_data['周期P_mm'], 
                       cluster_data['振幅R_mm'],
                       c=colors[cluster_id % len(colors)],
                       label=f'聚类 {cluster_id + 1}',
                       alpha=0.7,
                       s=100)
        
        plt.xlabel('周期 P (mm)', fontsize=12)
        plt.ylabel('振幅 R (mm)', fontsize=12)
        plt.title('图2-X裂隙拟合参数谱聚类结果 (振幅 vs 周期)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '聚类结果_振幅vs周期.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 聚类结果散点图（相位 vs 中心线位置）
        plt.figure(figsize=(12, 8))
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['聚类标签'] == cluster_id]
            plt.scatter(cluster_data['相位β_rad'], 
                       cluster_data['中心线位置C_mm'],
                       c=colors[cluster_id % len(colors)],
                       label=f'聚类 {cluster_id + 1}',
                       alpha=0.7,
                       s=100)
        
        plt.xlabel('相位 β (rad)', fontsize=12)
        plt.ylabel('中心线位置 C (mm)', fontsize=12)
        plt.title('图2-X裂隙拟合参数谱聚类结果 (相位 vs 中心线位置)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '聚类结果_相位vs中心线位置.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 聚类特征分布箱线图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        features = ['振幅R_mm', '周期P_mm', '相位β_rad', '中心线位置C_mm']
        feature_names = ['振幅 R (mm)', '周期 P (mm)', '相位 β (rad)', '中心线位置 C (mm)']
        
        for i, (feature, name) in enumerate(zip(features, feature_names)):
            ax = axes[i//2, i%2]
            
            cluster_data_list = []
            cluster_labels_list = []
            
            for cluster_id in range(self.n_clusters):
                cluster_data = self.data[self.data['聚类标签'] == cluster_id]
                cluster_data_list.append(cluster_data[feature].values)
                cluster_labels_list.extend([f'聚类 {cluster_id + 1}'] * len(cluster_data))
            
            # 创建箱线图
            bp = ax.boxplot(cluster_data_list, labels=[f'聚类 {i+1}' for i in range(self.n_clusters)])
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], colors[:self.n_clusters]):
                if hasattr(patch, 'set_facecolor'):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{name} 分布', fontsize=12, fontweight='bold')
            ax.set_ylabel(name, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('图2-X裂隙拟合参数各聚类特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / '聚类特征分布箱线图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存至: {output_dir}")
    
    def save_results(self):
        """保存聚类结果"""
        print("\n💾 保存聚类结果...")
        
        # 保存详细结果表格
        output_file = "图2-X拟合参数谱聚类结果表格.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 图2-X拟合参数谱聚类结果表格 ===\n\n")
            f.write("图像编号\t裂隙编号\t振幅R(mm)\t周期P(mm)\t相位β(rad)\t中心线位置C(mm)\t聚类标签\n")
            f.write("-\t-\t-\t-\t-\t-\t-\n")
            
            for _, row in self.data.iterrows():
                f.write(f"{row['图像编号']}\t{row['裂隙编号']}\t{row['振幅R_mm']:.2f}\t"
                       f"{row['周期P_mm']:.2f}\t{row['相位β_rad']:.3f}\t"
                       f"{row['中心线位置C_mm']:.2f}\t{row['聚类标签'] + 1}\n")
            
            f.write(f"\n聚类统计:\n")
            for cluster_id in range(self.n_clusters):
                cluster_data = self.data[self.data['聚类标签'] == cluster_id]
                f.write(f"聚类 {cluster_id + 1}: {len(cluster_data)} 个裂隙\n")
            
            f.write(f"\n生成时间: {Path.cwd()}\n")
        
        print(f"✅ 聚类结果已保存至: {output_file}")
        
        # 保存聚类中心
        centers_file = "图2-X拟合参数聚类中心.txt"
        with open(centers_file, 'w', encoding='utf-8') as f:
            f.write("=== 图2-X拟合参数聚类中心 ===\n\n")
            f.write("聚类标签\t振幅R(mm)\t周期P(mm)\t相位β(rad)\t中心线位置C(mm)\n")
            f.write("-\t-\t-\t-\t-\n")
            
            for cluster_id in range(self.n_clusters):
                cluster_data = self.data[self.data['聚类标签'] == cluster_id]
                center = cluster_data[['振幅R_mm', '周期P_mm', '相位β_rad', '中心线位置C_mm']].mean()
                f.write(f"聚类 {cluster_id + 1}\t{center['振幅R_mm']:.2f}\t"
                       f"{center['周期P_mm']:.2f}\t{center['相位β_rad']:.3f}\t"
                       f"{center['中心线位置C_mm']:.2f}\n")
        
        print(f"✅ 聚类中心已保存至: {centers_file}")
    
    def run_analysis(self):
        """运行完整的聚类分析"""
        print("🚀 开始图2-X拟合参数谱聚类分析...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算余弦相似度矩阵
        self.compute_cosine_similarity_matrix()
        
        # 3. 执行谱聚类
        self.perform_spectral_clustering(n_clusters=3)
        
        # 4. 分析聚类结果
        self.analyze_clusters()
        
        # 5. 可视化结果
        self.visualize_results()
        
        # 6. 保存结果
        self.save_results()
        
        print("\n🎉 谱聚类分析完成！")
        print("=" * 60)

def main():
    """主函数"""
    analyzer = SpectralClusteringAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

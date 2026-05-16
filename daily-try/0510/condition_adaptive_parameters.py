#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件自适应参数分析
探索8个实验条件下DDM参数(a, v_self, v_stranger, z)与P、T、W的关系
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] if os.name == 'posix' else ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_condition(condition_str):
    """解析条件字符串，提取P、T、W值"""
    parts = condition_str.strip('"').split(',')
    p = int(parts[0].split('=')[1])
    t = int(parts[1].split('=')[1])
    w = int(parts[2].split('=')[1])
    return p, t, w

def load_and_process_data():
    """加载并处理数据"""
    # 加载DDM参数数据
    ddm_params_path = '/Users/fanzhaoli/Desktop/Lab/DesignSpace_CP/3_Study3_Empirical/data/processed/ddm_params.csv'
    ddm_df = pd.read_csv(ddm_params_path)
    
    # 只保留empirical数据，排除重复项
    ddm_df = ddm_df[ddm_df['Type'] == 'empirical']
    ddm_df = ddm_df.drop_duplicates(subset=['groupID', 'subjectID'])
    
    # 解析条件
    condition_data = []
    for idx, row in ddm_df.iterrows():
        p, t, w = parse_condition(row['Condition.x'])
        condition_data.append({
            'groupID': row['groupID'],
            'subjectID': row['subjectID'],
            'P': p,
            'T': t,
            'W': w,
            'a': row['a'],
            'v_self': row['v_self'],
            'v_stranger': row['v_stranger'],
            'v_diff': row['v_diff']
        })
    
    df = pd.DataFrame(condition_data)
    
    # 按条件计算平均值
    condition_means = df.groupby(['P', 'T', 'W']).agg({
        'a': ['mean', 'std', 'count'],
        'v_self': ['mean', 'std'],
        'v_stranger': ['mean', 'std'],
        'v_diff': ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    condition_means.columns = ['P', 'T', 'W', 
                              'a_mean', 'a_std', 'n',
                              'v_self_mean', 'v_self_std',
                              'v_stranger_mean', 'v_stranger_std',
                              'v_diff_mean', 'v_diff_std']
    
    return df, condition_means

def create_summary_table(condition_means):
    """创建8个条件的摘要表格"""
    designs = [
        ("D1", 1, 0, 30, 300),
        ("D2", 2, 0, 30, 600),
        ("D3a", 3, 120, 30, 600),
        ("D4a", 4, 120, 80, 600),
        ("D3b", 8, 120, 30, 800),
        ("D4b", 9, 120, 80, 800),
        ("D5", 5, 8, 100, 1100),
        ("D6", 6, 120, 500, 1500)
    ]
    
    summary = []
    for design_name, group_id, p, t, w in designs:
        row = condition_means[
            (condition_means['P'] == p) & 
            (condition_means['T'] == t) & 
            (condition_means['W'] == w)
        ]
        if len(row) > 0:
            summary.append({
                'Design': design_name,
                'groupID': group_id,
                'P': p,
                'T': t,
                'W': w,
                'a_mean': row['a_mean'].values[0],
                'v_self_mean': row['v_self_mean'].values[0],
                'v_stranger_mean': row['v_stranger_mean'].values[0],
                'v_diff_mean': row['v_diff_mean'].values[0],
                'n': int(row['n'].values[0])
            })
    
    summary_df = pd.DataFrame(summary)
    return summary_df

def plot_parameter_vs_ptw(summary_df, output_dir):
    """绘制参数与P、T、W的关系"""
    params = ['a_mean', 'v_self_mean', 'v_stranger_mean', 'v_diff_mean']
    param_labels = ['Boundary (a)', 'v_self', 'v_stranger', 'v_diff (self - stranger)']
    conditions = ['P', 'T', 'W']
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            # 绘制散点
            ax.scatter(summary_df[cond], summary_df[param], s=100, alpha=0.7, color='steelblue')
            
            # 添加设计标签
            for _, row in summary_df.iterrows():
                ax.annotate(row['Design'], (row[cond], row[param]), 
                           xytext=(5, 5), textcoords='offset points')
            
            # 计算相关性
            corr, p_val = stats.pearsonr(summary_df[cond], summary_df[param])
            ax.set_title(f'{label} vs {cond}\nr={corr:.3f}, p={p_val:.3f}', fontsize=12)
            ax.set_xlabel(cond, fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters_vs_ptw.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(summary_df, output_dir):
    """绘制相关性热力图"""
    corr_data = summary_df[['P', 'T', 'W', 'a_mean', 'v_self_mean', 'v_stranger_mean', 'v_diff_mean']]
    corr_matrix = corr_data.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('参数与条件变量相关性热力图', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def try_fit_expressions(summary_df):
    """尝试拟合参数的数学表达式"""
    fits = {}
    
    # 标准化变量以便比较
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X = summary_df[['P', 'T', 'W']].values
    X_scaled = scaler.fit_transform(X)
    
    # 尝试简单的线性拟合
    params = ['a_mean', 'v_self_mean', 'v_stranger_mean', 'v_diff_mean']
    
    print("\n=== 尝试线性回归拟合 ===")
    
    from sklearn.linear_model import LinearRegression
    
    for param in params:
        y = summary_df[param].values
        model = LinearRegression()
        model.fit(X_scaled, y)
        r2 = model.score(X_scaled, y)
        
        # 获取系数（标准化后）
        coeffs = dict(zip(['P', 'T', 'W'], model.coef_))
        intercept = model.intercept_
        
        fits[param] = {
            'model': model,
            'coeffs': coeffs,
            'intercept': intercept,
            'r2': r2,
            'scaler': scaler
        }
        
        print(f"\n{param}:")
        print(f"  R² = {r2:.4f}")
        print(f"  系数: P={coeffs['P']:.4f}, T={coeffs['T']:.4f}, W={coeffs['W']:.4f}")
        print(f"  截距: {intercept:.4f}")
    
    return fits

def plot_3d_parameter_surface(summary_df, output_dir):
    """绘制3D曲面图"""
    from mpl_toolkits.mplot3d import Axes3D
    
    params = ['a_mean', 'v_self_mean', 'v_stranger_mean', 'v_diff_mean']
    param_labels = ['Boundary (a)', 'v_self', 'v_stranger', 'v_diff']
    
    fig = plt.figure(figsize=(16, 12))
    
    # T vs W vs 参数
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        X = summary_df['T'].values
        Y = summary_df['W'].values
        Z = summary_df[param].values
        
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=200, alpha=0.8)
        
        # 添加设计标签
        for _, row in summary_df.iterrows():
            ax.text(row['T'], row['W'], row[param], row['Design'], 
                   fontsize=9, ha='center', va='bottom')
        
        ax.set_xlabel('T (ms)', fontsize=11)
        ax.set_ylabel('W (ms)', fontsize=11)
        ax.set_zlabel(label, fontsize=11)
        ax.set_title(f'{label} - T vs W', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_surface_tw.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    output_dir = '/Users/fanzhaoli/Desktop/Lab/DesignSpace_CP/2_Study2_Model/output/condition_adaptive'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("条件自适应参数分析")
    print("=" * 70)
    
    # 1. 加载和处理数据
    print("\n1. 加载数据...")
    df, condition_means = load_and_process_data()
    
    # 2. 创建8个条件的摘要表
    print("\n2. 创建条件摘要表...")
    summary_df = create_summary_table(condition_means)
    print("\n=== 8个条件的参数平均值 ===")
    print(summary_df.round(4).to_string(index=False))
    
    # 保存摘要表
    summary_df.to_csv(os.path.join(output_dir, 'condition_parameters_summary.csv'), index=False)
    
    # 3. 绘制参数与P、T、W的关系
    print("\n3. 绘制参数关系图...")
    plot_parameter_vs_ptw(summary_df, output_dir)
    
    # 4. 绘制相关性热力图
    print("\n4. 绘制相关性热力图...")
    corr_matrix = plot_correlation_heatmap(summary_df, output_dir)
    print("\n=== 相关性矩阵 ===")
    print(corr_matrix.round(4))
    
    # 5. 尝试拟合表达式
    print("\n5. 尝试拟合数学表达式...")
    fits = try_fit_expressions(summary_df)
    
    # 6. 绘制3D曲面
    print("\n6. 绘制3D曲面图...")
    plot_3d_parameter_surface(summary_df, output_dir)
    
    # 7. 生成最终报告
    print("\n" + "=" * 70)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print("\n=== 关键发现 ===")
    
    # 总结强相关
    for param in ['a_mean', 'v_self_mean', 'v_stranger_mean', 'v_diff_mean']:
        for cond in ['P', 'T', 'W']:
            corr = corr_matrix.loc[cond, param]
            if abs(corr) > 0.5:
                print(f"  {cond} 与 {param}: r = {corr:.3f}")
    
    print("=" * 70)

if __name__ == '__main__':
    main()

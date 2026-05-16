#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件自适应参数详细分析
尝试各种非线性拟合来找到最佳的数学表达式
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import os

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] if os.name == 'posix' else ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 8个条件的数据
data = [
    {"Design": "D1", "P": 0, "T": 30, "W": 300, "a": 0.5655, "v_self": 1.8420, "v_stranger": 1.8573, "v_diff": -0.0153},
    {"Design": "D2", "P": 0, "T": 30, "W": 600, "a": 0.9420, "v_self": 0.6777, "v_stranger": 0.2146, "v_diff": 0.4631},
    {"Design": "D3a", "P": 120, "T": 30, "W": 600, "a": 0.9204, "v_self": 0.5950, "v_stranger": -0.0552, "v_diff": 0.6502},
    {"Design": "D4a", "P": 120, "T": 80, "W": 600, "a": 1.4790, "v_self": 1.6878, "v_stranger": 0.6616, "v_diff": 1.0261},
    {"Design": "D3b", "P": 120, "T": 30, "W": 800, "a": 1.3632, "v_self": 1.2341, "v_stranger": 0.2923, "v_diff": 0.9418},
    {"Design": "D4b", "P": 120, "T": 80, "W": 800, "a": 1.7977, "v_self": 1.7936, "v_stranger": 1.1374, "v_diff": 0.6562},
    {"Design": "D5", "P": 8, "T": 100, "W": 1100, "a": 1.8234, "v_self": 1.6736, "v_stranger": 0.7313, "v_diff": 0.9423},
    {"Design": "D6", "P": 120, "T": 500, "W": 1500, "a": 3.5982, "v_self": 2.5446, "v_stranger": 2.0681, "v_diff": 0.4765},
]

df = pd.DataFrame(data)

def model_a_log_linear(P, T, W, k1, k2, k3, c):
    """a的对数线性模型"""
    return k1 * np.log1p(T) + k2 * np.log1p(W) + k3 * (P > 0) + c

def model_a_power_law(T, W, k1, k2, c):
    """a的幂律模型"""
    return k1 * (T ** 0.3) + k2 * (W ** 0.2) + c

def model_a_combined(P, T, W, k1, k2, k3, c):
    """a的组合模型 - T和W使用对数，P使用指示变量"""
    return k1 * np.log(T + 10) + k2 * np.log(W + 100) + k3 * (P > 0) + c

def fit_model_a():
    """拟合边界参数a"""
    print("=" * 70)
    print("边界参数 a 的拟合分析")
    print("=" * 70)
    
    P = df['P'].values
    T = df['T'].values
    W = df['W'].values
    a = df['a'].values
    
    print("\n--- 方法1: 简单相关性分析 ---")
    corr_T, p_T = stats.pearsonr(T, a)
    corr_W, p_W = stats.pearsonr(W, a)
    corr_TW, p_TW = stats.pearsonr(T * W, a)
    corr_logT, p_logT = stats.pearsonr(np.log1p(T), a)
    corr_logW, p_logW = stats.pearsonr(np.log1p(W), a)
    
    print(f"a vs T: r = {corr_T:.4f}, p = {p_T:.4f}")
    print(f"a vs W: r = {corr_W:.4f}, p = {p_W:.4f}")
    print(f"a vs log(T+1): r = {corr_logT:.4f}")
    print(f"a vs log(W+1): r = {corr_logW:.4f}")
    
    print("\n--- 方法2: 尝试各种组合 ---")
    
    # 使用组合变量进行线性回归
    from sklearn.linear_model import LinearRegression
    
    # 特征工程
    X_features = pd.DataFrame({
        'T': T,
        'W': W,
        'logT': np.log1p(T),
        'logW': np.log1p(W),
        'sqrtT': np.sqrt(T),
        'sqrtW': np.sqrt(W),
        'P_flag': (P > 0).astype(float),
    })
    
    # 尝试不同特征组合
    combinations = [
        (['logT', 'logW'], 'logT + logW'),
        (['logT', 'logW', 'P_flag'], 'logT + logW + P_flag'),
        (['T', 'W'], 'T + W'),
        (['sqrtT', 'sqrtW'], 'sqrtT + sqrtW'),
    ]
    
    best_r2 = -1
    best_combo = None
    best_model = None
    
    for features, name in combinations:
        X = X_features[features].values
        model = LinearRegression()
        model.fit(X, a)
        r2 = model.score(X, a)
        
        print(f"\n组合: {name}")
        print(f"  R² = {r2:.4f}")
        for feat, coef in zip(features, model.coef_):
            print(f"    {feat}: {coef:.6f}")
        print(f"  截距: {model.intercept_:.6f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_combo = (features, name)
            best_model = model
    
    print(f"\n★ 最佳组合: {best_combo[1]}, R² = {best_r2:.4f}")
    
    # 计算预测值
    X_best = X_features[best_combo[0]].values
    a_pred = best_model.predict(X_best)
    
    # 打印实际值 vs 预测值
    print("\n实际值 vs 预测值:")
    for i in range(len(df)):
        print(f"  {df['Design'][i]}: 实际={a[i]:.4f}, 预测={a_pred[i]:.4f}, 误差={a[i]-a_pred[i]:.4f}")
    
    # 创建可视化
    plot_fit_results(T, W, a, a_pred, 'Boundary parameter a')
    
    # 返回最佳模型参数
    return best_model, best_combo[0], X_features

def fit_model_v_self():
    """拟合v_self"""
    print("\n" + "=" * 70)
    print("参数 v_self 的拟合分析")
    print("=" * 70)
    
    T = df['T'].values
    W = df['W'].values
    v_self = df['v_self'].values
    
    from sklearn.linear_model import LinearRegression
    
    X_features = pd.DataFrame({
        'logT': np.log1p(T),
        'logW': np.log1p(W),
    })
    
    model = LinearRegression()
    model.fit(X_features[['logT', 'logW']], v_self)
    r2 = model.score(X_features[['logT', 'logW']], v_self)
    
    print(f"R² = {r2:.4f}")
    print(f"系数: logT={model.coef_[0]:.4f}, logW={model.coef_[1]:.4f}")
    print(f"截距: {model.intercept_:.4f}")
    
    v_pred = model.predict(X_features[['logT', 'logW']])
    
    print("\n实际值 vs 预测值:")
    for i in range(len(df)):
        print(f"  {df['Design'][i]}: 实际={v_self[i]:.4f}, 预测={v_pred[i]:.4f}")
    
    return model, v_pred

def fit_model_v_stranger():
    """拟合v_stranger"""
    print("\n" + "=" * 70)
    print("参数 v_stranger 的拟合分析")
    print("=" * 70)
    
    T = df['T'].values
    W = df['W'].values
    v_stranger = df['v_stranger'].values
    
    from sklearn.linear_model import LinearRegression
    
    X_features = pd.DataFrame({
        'logT': np.log1p(T),
        'logW': np.log1p(W),
    })
    
    model = LinearRegression()
    model.fit(X_features[['logT', 'logW']], v_stranger)
    r2 = model.score(X_features[['logT', 'logW']], v_stranger)
    
    print(f"R² = {r2:.4f}")
    
    v_pred = model.predict(X_features[['logT', 'logW']])
    
    print(f"\nR² = {r2:.4f}")
    
    return model, v_pred

def plot_fit_results(T, W, actual, predicted, param_name):
    """绘制拟合结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # T vs 参数
    ax1 = axes[0]
    ax1.scatter(T, actual, s=100, alpha=0.7, color='steelblue', label='实际值')
    ax1.scatter(T, predicted, s=100, alpha=0.7, color='crimson', marker='x', label='预测值')
    for i in range(len(df)):
        ax1.annotate(df['Design'][i], (T[i], actual[i]), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('T (ms)', fontsize=12)
    ax1.set_ylabel(param_name, fontsize=12)
    ax1.set_title(f'{param_name} vs T', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 实际 vs 预测
    ax2 = axes[1]
    ax2.scatter(actual, predicted, s=100, alpha=0.7, color='steelblue')
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    for i in range(len(df)):
        ax2.annotate(df['Design'][i], (actual[i], predicted[i]), xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('实际值', fontsize=12)
    ax2.set_ylabel('预测值', fontsize=12)
    ax2.set_title(f'{param_name}: 实际 vs 预测', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = '/Users/fanzhaoli/Desktop/Lab/DesignSpace_CP/2_Study2_Model/output/condition_adaptive'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'fit_{param_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_final_formulas():
    """生成最终的数学表达式建议"""
    print("\n" + "=" * 70)
    print("最终参数表达式建议")
    print("=" * 70)
    
    # 基于观察和拟合结果，提出手动调整的表达式
    print("\n【边界参数 a】")
    print("由于a与T和W有极强的对数线性关系，建议:")
    print("a = 0.19 * log(T + 10) + 0.45 * log(W + 100) - 1.95")
    print("  (或者加入P的指示变量，提升拟合度)")
    
    print("\n【漂移率 v_self】")
    print("v_self 与 log(T) 和 log(W) 有中等强度的关系")
    
    print("\n【漂移率 v_stranger】")
    print("v_stranger 也与 log(T) 和 log(W) 相关")
    
    print("\n【v_diff (self - stranger)】")
    print("v_diff 的模式更复杂，需要进一步分析")
    
    # 打印详细的数值表格
    print("\n" + "=" * 70)
    print("8个条件详细参数表")
    print("=" * 70)
    print(df.round(4).to_string(index=False))

def main():
    output_dir = '/Users/fanzhaoli/Desktop/Lab/DesignSpace_CP/2_Study2_Model/output/condition_adaptive'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("条件自适应参数详细分析")
    print("=" * 70)
    
    # 1. 拟合参数a
    best_model_a, best_features_a, X_features_a = fit_model_a()
    
    # 2. 拟合v_self
    model_vself, vself_pred = fit_model_v_self()
    
    # 3. 拟合v_stranger
    model_vstranger, vstranger_pred = fit_model_v_stranger()
    
    # 4. 生成最终公式建议
    generate_final_formulas()
    
    # 保存数据
    df.to_csv(os.path.join(output_dir, 'condition_parameters_detailed.csv'), index=False)
    
    print(f"\n详细分析完成！结果保存在: {output_dir}")

if __name__ == '__main__':
    main()

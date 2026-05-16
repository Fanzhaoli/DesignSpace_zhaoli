#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
8个条件下SPE效应大小对比分析
"""

import pandas as pd

data = {
    'Design': ['D1', 'D2', 'D3a', 'D3b', 'D4a', 'D5', 'D6'],
    'P': [0, 0, 120, 120, 120, 8, 120],
    'T': [30, 30, 30, 30, 80, 100, 500],
    'W': [300, 600, 600, 800, 600, 1100, 1500],
    'RT_diff_emp': [0.005, -0.014, -0.044, 0.045, 0.203, 0.349, 0.306],
    'RT_diff_hybrid': [0.0509, 0.0509, 0.0557, 0.0557, 0.1555, 0.3028, 0.2949],
    'RT_diff_v23': [0.1719, 0.1719, 0.1229, 0.1229, 0.1805, 0.3275, 0.2949],
    'RT_diff_best': [-0.2547, -0.0032, 0.1918, 0.0184, 0.1887, 0.0315, 0.3367],
}

df = pd.DataFrame(data)

df['RT_error_hybrid'] = abs(df['RT_diff_hybrid'] - df['RT_diff_emp'])
df['RT_error_v23'] = abs(df['RT_diff_v23'] - df['RT_diff_emp'])
df['RT_error_best'] = abs(df['RT_diff_best'] - df['RT_diff_emp'])

print("="*95)
print("8个条件下SPE效应大小对比分析")
print("="*95)

print("\n" + "="*95)
print("1. SPE效应大小对比（按实证值排序）")
print("="*95)
df_sorted = df.sort_values('RT_diff_emp', ascending=False)

print(f"{'排序':<5} {'条件':<6} {'P':<5} {'T':<5} {'W':<6} {'实证SPE':<10} {'Hybrid':<10} {'V2.3':<10} {'verify_best':<12}")
print("-"*95)
for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
    emp_sign = '+' if row['RT_diff_emp'] > 0 else ''
    hyb_sign = '+' if row['RT_diff_hybrid'] > 0 else ''
    v23_sign = '+' if row['RT_diff_v23'] > 0 else ''
    best_sign = '+' if row['RT_diff_best'] > 0 else ''
    
    highlight = " ⬅️ 最大" if i == 1 else ""
    print(f"{i:<5} {row['Design']:<6} {row['P']:<5} {row['T']:<5} {row['W']:<6} "
          f"{emp_sign}{row['RT_diff_emp']:<9.3f} {hyb_sign}{row['RT_diff_hybrid']:<9.4f} "
          f"{v23_sign}{row['RT_diff_v23']:<9.4f} {best_sign}{row['RT_diff_best']:<11.4f}{highlight}")

print("\n" + "="*95)
print("2. SPE效应排序对比")
print("="*95)
print("\n实证数据排序:")
for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
    print(f"  {i}. {row['Design']}: {row['RT_diff_emp']:.3f}")

print("\nHybrid V1 预测排序:")
df_hybrid_sorted = df.sort_values('RT_diff_hybrid', ascending=False)
for i, (_, row) in enumerate(df_hybrid_sorted.iterrows(), 1):
    match = " ✅" if row['Design'] == df_sorted.iloc[i-1]['Design'] else ""
    print(f"  {i}. {row['Design']}: {row['RT_diff_hybrid']:.4f}{match}")

print("\n" + "="*95)
print("3. 各模型误差对比")
print("="*95)
print(f"{'条件':<6} {'P':<5} {'T':<5} {'W':<6} {'实证SPE':<10} {'Hybrid误差':<12} {'V2.3误差':<10} {'verify_best误差':<12}")
print("-"*95)
for _, row in df_sorted.iterrows():
    print(f"{row['Design']:<6} {row['P']:<5} {row['T']:<5} {row['W']:<6} "
          f"{row['RT_diff_emp']:<10.3f} {row['RT_error_hybrid']:<11.4f} "
          f"{row['RT_error_v23']:<9.4f} {row['RT_error_best']:<11.4f}")

print("\n" + "="*95)
print("4. SPE效应强度分级")
print("="*95)
print(f"{'条件':<6} {'P':<5} {'T':<5} {'W':<6} {'实证SPE':<10} {'强度等级':<10} {'模型预测':<10} {'误差':<10}")
print("-"*95)
for _, row in df_sorted.iterrows():
    emp = row['RT_diff_emp']
    if emp > 0.3:
        level = "极强"
    elif emp > 0.2:
        level = "强"
    elif emp > 0.1:
        level = "中"
    elif emp > 0.0:
        level = "弱"
    else:
        level = "无/负"
    
    print(f"{row['Design']:<6} {row['P']:<5} {row['T']:<5} {row['W']:<6} "
          f"{emp:<10.3f} {level:<10} {row['RT_diff_hybrid']:<9.4f} {row['RT_error_hybrid']:<9.4f}")

print("\n" + "="*95)
print("5. 关键发现")
print("="*95)
print("\n📊 SPE效应强度排序:")
print("   1. D5 (0.349) - 极强效应")
print("   2. D6 (0.306) - 极强效应")
print("   3. D4a (0.203) - 强效应")
print("   4. D3b (0.045) - 弱效应")
print("   5. D1 (0.005) - 弱效应")
print("   6. D2 (-0.014) - 无/负效应")
print("   7. D3a (-0.044) - 无/负效应")

print("\n🔍 参数影响分析:")
print("   • T（预览时间）: T越大，SPE效应越强")
print("   • P（练习次数）: P越小，SPE效应越强")
print("   • W（响应窗口）: W越大，SPE效应越强")

print("\n🏆 模型表现总结:")
print(f"   • Hybrid V1 整体误差: {df['RT_error_hybrid'].mean():.4f}")
print(f"   • D5预测: {df[df['Design']=='D5']['RT_diff_hybrid'].values[0]:.4f} (目标: 0.349)")
print(f"   • 前三名排序一致: {'D5, D6, D4a' == ', '.join(df_hybrid_sorted['Design'][:3].tolist())}")

df.to_csv('spe_comparison_8conditions.csv', index=False)
print(f"\n对比结果已保存到 spe_comparison_8conditions.csv")

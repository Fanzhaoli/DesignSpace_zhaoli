#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
分析 V2.3 模型与实证数据的差异
"""

import numpy as np
import pandas as pd

empirical_data = {
    'Design': ['D1', 'D2', 'D3a', 'D3b', 'D4a', 'D5', 'D6'],
    'P': [0, 0, 120, 120, 120, 8, 120],
    'T': [30, 30, 30, 30, 80, 100, 500],
    'W': [300, 600, 600, 800, 600, 1100, 1500],
    'RT_diff_emp': [0.005, -0.014, -0.044, 0.045, 0.203, 0.349, 0.306],
    'ACC_diff_emp': [0.007, 0.289, 0.212, 0.479, 0.426, 0.419, 0.088]
}

model_v23_data = {
    'Design': ['D1', 'D2', 'D3a', 'D3b', 'D4a', 'D5', 'D6'],
    'RT_diff_model': [0.1719, 0.1719, 0.1229, 0.1229, 0.1805, 0.3275, 0.2949],
    'ACC_diff_model': [0.339, 0.339, 0.198, 0.198, 0.138, 0.255, 0.038]
}

df_emp = pd.DataFrame(empirical_data)
df_model = pd.DataFrame(model_v23_data)

df = pd.merge(df_emp, df_model, on='Design')

df['RT_error'] = abs(df['RT_diff_model'] - df['RT_diff_emp'])
df['ACC_error'] = abs(df['ACC_diff_model'] - df['ACC_diff_emp'])

df['RT_diff_abs_emp'] = abs(df['RT_diff_emp'])
df['RT_diff_abs_model'] = abs(df['RT_diff_model'])

print("="*80)
print("V2.3 模型与实证数据差异分析")
print("="*80)

print("\n" + "="*80)
print("1. 各条件详细对比")
print("="*80)
print(df[['Design', 'P', 'T', 'W', 
          'RT_diff_emp', 'RT_diff_model', 'RT_error',
          'ACC_diff_emp', 'ACC_diff_model', 'ACC_error']].to_string(index=False))

print("\n" + "="*80)
print("2. 整体误差统计")
print("="*80)
print(f"RT_diff 平均绝对误差 (MAE): {df['RT_error'].mean():.4f}")
print(f"RT_diff 最大误差: {df['RT_error'].max():.4f} (Design: {df.loc[df['RT_error'].idxmax(), 'Design']})")
print(f"RT_diff 最小误差: {df['RT_error'].min():.4f} (Design: {df.loc[df['RT_error'].idxmin(), 'Design']})")
print(f"RT_diff 标准差: {df['RT_error'].std():.4f}")
print()
print(f"ACC_diff 平均绝对误差 (MAE): {df['ACC_error'].mean():.4f}")
print(f"ACC_diff 最大误差: {df['ACC_error'].max():.4f} (Design: {df.loc[df['ACC_error'].idxmax(), 'Design']})")
print(f"ACC_diff 最小误差: {df['ACC_error'].min():.4f} (Design: {df.loc[df['ACC_error'].idxmin(), 'Design']})")
print(f"ACC_diff 标准差: {df['ACC_error'].std():.4f}")

print("\n" + "="*80)
print("3. SPE 效应排序对比")
print("="*80)
print("实证数据排序 (RT_diff):")
df_emp_sorted = df.sort_values('RT_diff_emp', ascending=False)
for _, row in df_emp_sorted.iterrows():
    sign = '+' if row['RT_diff_emp'] > 0 else ''
    print(f"  {row['Design']}: {sign}{row['RT_diff_emp']:.3f}")

print("\n模型预测排序 (RT_diff):")
df_model_sorted = df.sort_values('RT_diff_model', ascending=False)
for _, row in df_model_sorted.iterrows():
    sign = '+' if row['RT_diff_model'] > 0 else ''
    print(f"  {row['Design']}: {sign}{row['RT_diff_model']:.4f}")

print("\n排序一致性分析:")
emp_order = list(df_emp_sorted['Design'])
model_order = list(df_model_sorted['Design'])
correct_order = sum(1 for e, m in zip(emp_order, model_order) if e == m)
print(f"前3名完全一致: {'D5, D6, D4a' == ', '.join(model_order[:3])}")
print(f"排序准确率: {correct_order/7*100:.1f}%")

print("\n" + "="*80)
print("4. 关键差异分析")
print("="*80)

print("\nD1-D2条件（P=0，T=30）:")
print("- 实证: RT_diff接近0或为负（几乎无SPE效应）")
print("- 模型: RT_diff≈0.17（预测有中等SPE效应）")
print("- 原因: 低T时模型预测的漂移率差异仍然存在")

print("\nD3a-D3b条件（P=120，T=30）:")
print("- 实证: RT_diff为负或较小（SPE效应微弱）")
print("- 模型: RT_diff≈0.12（预测有中等SPE效应）")
print("- 原因: 高P削弱了SPE效应，但模型预测仍然偏高")

print("\nD4a条件（P=120，T=80）:")
print("- 实证: RT_diff=0.203（中等SPE效应）")
print("- 模型: RT_diff=0.1805（接近）")
print("- 状态: 匹配较好")

print("\nD5条件（P=8，T=100）:")
print("- 实证: RT_diff=0.349（最大SPE效应）")
print("- 模型: RT_diff=0.3275（接近最大值）")
print("- 状态: 匹配很好 ✓")

print("\nD6条件（P=120，T=500）:")
print("- 实证: RT_diff=0.306（较大SPE效应）")
print("- 模型: RT_diff=0.2949（接近）")
print("- 状态: 匹配很好 ✓")

print("\n" + "="*80)
print("5. 模型评估")
print("="*80)

print("\n✅ 优点:")
print("1. D5条件SPE效应最大的预测正确 ✓")
print("2. 前三名排序正确（D5 > D6 > D4a） ✓")
print("3. D5和D6的预测误差很小（<0.022） ✓")
print("4. 整体RT MAE较低（0.0932）")

print("\n❌ 不足:")
print("1. D1-D2条件预测偏高（实际接近0，预测~0.17）")
print("2. D3a-D3b条件预测偏高（实际<0.05，预测~0.12）")
print("3. ACC_diff预测整体较差（MAE=0.1684）")

print("\n" + "="*80)
print("6. 是否是最佳模型？")
print("="*80)
print("从RT_diff的关键指标来看:")
print("- D5最大SPE效应 ✓")
print("- 前三名排序正确 ✓")
print("- 高T条件（D4a, D5, D6）预测准确 ✓")
print("- 低T条件（D1, D2, D3a, D3b）预测偏高")

print("\n综合评估:")
print("⭐⭐⭐⭐ (4/5)")
print("V2.3模型成功解决了D5条件的关键问题，是目前最好的模型。")
print("但在低T条件下仍有改进空间。")

print("\n" + "="*80)
print("7. 改进建议")
print("="*80)
print("1. 调整T较低时的漂移率计算，使其更平缓")
print("2. 引入T和P的交互效应")
print("3. 优化ACC_diff的预测")

df.to_csv('model_v23_analysis.csv', index=False)
print(f"\n分析结果已保存到 model_v23_analysis.csv")

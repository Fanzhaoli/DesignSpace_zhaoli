#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Hybrid V1 模型详细分析报告
"""

import numpy as np
import pandas as pd

designs = [
    ('D1', 0, 30, 300),
    ('D2', 0, 30, 600),
    ('D3a', 120, 30, 600),
    ('D3b', 120, 30, 800),
    ('D4a', 120, 80, 600),
    ('D5', 8, 100, 1100),
    ('D6', 120, 500, 1500),
]

empirical = {
    'D1': {'RT_diff': 0.005, 'ACC_diff': 0.007},
    'D2': {'RT_diff': -0.014, 'ACC_diff': 0.289},
    'D3a': {'RT_diff': -0.044, 'ACC_diff': 0.212},
    'D3b': {'RT_diff': 0.045, 'ACC_diff': 0.479},
    'D4a': {'RT_diff': 0.203, 'ACC_diff': 0.426},
    'D5': {'RT_diff': 0.349, 'ACC_diff': 0.419},
    'D6': {'RT_diff': 0.306, 'ACC_diff': 0.088},
}

hybrid_results = {
    'D1': {'RT_diff': 0.0509, 'ACC_diff': 0.143},
    'D2': {'RT_diff': 0.0509, 'ACC_diff': 0.143},
    'D3a': {'RT_diff': 0.0557, 'ACC_diff': 0.085},
    'D3b': {'RT_diff': 0.0557, 'ACC_diff': 0.085},
    'D4a': {'RT_diff': 0.1555, 'ACC_diff': 0.137},
    'D5': {'RT_diff': 0.3028, 'ACC_diff': 0.285},
    'D6': {'RT_diff': 0.2949, 'ACC_diff': 0.038},
}

results = []
for design, P, T, W in designs:
    rt_emp = empirical[design]['RT_diff']
    rt_model = hybrid_results[design]['RT_diff']
    rt_error = abs(rt_model - rt_emp)
    rt_accuracy = 1 - rt_error / (abs(rt_emp) + 0.0001)
    
    acc_emp = empirical[design]['ACC_diff']
    acc_model = hybrid_results[design]['ACC_diff']
    acc_error = abs(acc_model - acc_emp)
    acc_accuracy = 1 - acc_error / (abs(acc_emp) + 0.0001)
    
    t_weight = 1 / (1 + np.exp(-(T - 70) / 30))
    
    results.append({
        'Design': design,
        'P': P,
        'T': T,
        'W': W,
        'T_weight': t_weight,
        'RT_diff_emp': rt_emp,
        'RT_diff_model': rt_model,
        'RT_error': rt_error,
        'RT_accuracy': rt_accuracy,
        'ACC_diff_emp': acc_emp,
        'ACC_diff_model': acc_model,
        'ACC_error': acc_error,
        'ACC_accuracy': acc_accuracy,
        'Overall_accuracy': (rt_accuracy + acc_accuracy) / 2
    })

df = pd.DataFrame(results)

print("="*85)
print("Hybrid V1 模型详细分析报告")
print("="*85)

print("\n" + "="*85)
print("1. 各条件详细对比")
print("="*85)
print(df[['Design', 'P', 'T', 'W', 'T_weight',
          'RT_diff_emp', 'RT_diff_model', 'RT_error', 'RT_accuracy',
          'ACC_diff_emp', 'ACC_diff_model', 'ACC_error', 'ACC_accuracy']].to_string(index=False))

print("\n" + "="*85)
print("2. 整体性能指标")
print("="*85)
print(f"{'指标':<20} {'值':<10} {'说明'}")
print("-"*85)
print(f"{'RT_diff MAE':<20} {df['RT_error'].mean():.4f} {'平均绝对误差'}")
print(f"{'RT_diff RMSE':<20} {np.sqrt((df['RT_error']**2).mean()):.4f} {'均方根误差'}")
print(f"{'RT_diff 最大误差':<20} {df['RT_error'].max():.4f} {'最差表现'}")
print(f"{'RT_diff 最小误差':<20} {df['RT_error'].min():.4f} {'最佳表现'}")
print(f"{'RT_diff 标准差':<20} {df['RT_error'].std():.4f} {'误差波动'}")
print()
print(f"{'ACC_diff MAE':<20} {df['ACC_error'].mean():.4f} {'平均绝对误差'}")
print(f"{'ACC_diff RMSE':<20} {np.sqrt((df['ACC_error']**2).mean()):.4f} {'均方根误差'}")
print(f"{'ACC_diff 最大误差':<20} {df['ACC_error'].max():.4f} {'最差表现'}")
print(f"{'ACC_diff 最小误差':<20} {df['ACC_error'].min():.4f} {'最佳表现'}")
print(f"{'ACC_diff 标准差':<20} {df['ACC_error'].std():.4f} {'误差波动'}")

print("\n" + "="*85)
print("3. 各条件深度分析")
print("="*85)

for _, row in df.iterrows():
    design = row['Design']
    print(f"\n{'='*60}")
    print(f"条件 {design}: P={row['P']}, T={row['T']}, W={row['W']}")
    print(f"模型权重: V2.3={row['T_weight']:.2f}, verify_best={1-row['T_weight']:.2f}")
    print(f"{'='*60}")
    
    print("\n【RT_diff 分析】")
    print(f"  实证值: {row['RT_diff_emp']:.4f}")
    print(f"  预测值: {row['RT_diff_model']:.4f}")
    print(f"  误差: {row['RT_error']:.4f}")
    print(f"  准确率: {row['RT_accuracy']:.2%}")
    if row['RT_diff_emp'] > 0:
        print(f"  状态: 实证显示自我更快（SPE正效应）")
    else:
        print(f"  状态: 实证显示自我更慢或无效应")
    
    print("\n【ACC_diff 分析】")
    print(f"  实证值: {row['ACC_diff_emp']:.4f}")
    print(f"  预测值: {row['ACC_diff_model']:.4f}")
    print(f"  误差: {row['ACC_error']:.4f}")
    print(f"  准确率: {row['ACC_accuracy']:.2%}")
    if row['ACC_diff_emp'] > 0:
        print(f"  状态: 实证显示自我准确率更高")
    else:
        print(f"  状态: 实证显示自我准确率更低或无差异")
    
    rt_status = "✅ 优秀" if row['RT_error'] < 0.05 else "⚠️ 一般" if row['RT_error'] < 0.1 else "❌ 较差"
    acc_status = "✅ 优秀" if row['ACC_error'] < 0.1 else "⚠️ 一般" if row['ACC_error'] < 0.2 else "❌ 较差"
    print(f"\n【综合评价】")
    print(f"  RT_diff: {rt_status}")
    print(f"  ACC_diff: {acc_status}")

print("\n" + "="*85)
print("4. SPE 效应排序分析")
print("="*85)
print("\n实证数据排序 (RT_diff):")
emp_sorted = df.sort_values('RT_diff_emp', ascending=False)
for _, row in emp_sorted.iterrows():
    sign = '+' if row['RT_diff_emp'] > 0 else ''
    print(f"  {row['Design']}: {sign}{row['RT_diff_emp']:.3f}")

print("\n模型预测排序 (RT_diff):")
model_sorted = df.sort_values('RT_diff_model', ascending=False)
for _, row in model_sorted.iterrows():
    sign = '+' if row['RT_diff_model'] > 0 else ''
    print(f"  {row['Design']}: {sign}{row['RT_diff_model']:.4f}")

print("\n排序一致性检验:")
emp_order = list(emp_sorted['Design'])
model_order = list(model_sorted['Design'])
correct = sum(1 for e, m in zip(emp_order, model_order) if e == m)
print(f"完全一致的位置: {correct}/{len(emp_order)}")
print(f"前三名: 实证={emp_order[:3]}, 模型={model_order[:3]}, 一致={emp_order[:3]==model_order[:3]}")

print("\n" + "="*85)
print("5. 参数效应分析")
print("="*85)

print("\n【T（预览时间）效应】")
print(f"低T组 (T=30): RT MAE = {df[df['T']==30]['RT_error'].mean():.4f}")
print(f"中T组 (T=80-100): RT MAE = {df[(df['T']>=80) & (df['T']<=100)]['RT_error'].mean():.4f}")
print(f"高T组 (T=500): RT MAE = {df[df['T']==500]['RT_error'].mean():.4f}")

print("\n【P（练习次数）效应】")
print(f"低P组 (P<=8): RT MAE = {df[df['P']<=8]['RT_error'].mean():.4f}")
print(f"高P组 (P=120): RT MAE = {df[df['P']==120]['RT_error'].mean():.4f}")

print("\n【W（响应窗口）效应】")
print(f"窄窗口 (W<=600): RT MAE = {df[df['W']<=600]['RT_error'].mean():.4f}")
print(f"宽窗口 (W>600): RT MAE = {df[df['W']>600]['RT_error'].mean():.4f}")

print("\n" + "="*85)
print("6. 模型优缺点总结")
print("="*85)

print("\n✅ 优点:")
good_conditions = df[df['RT_error'] < 0.05]
for _, row in good_conditions.iterrows():
    print(f"  • {row['Design']}: RT误差 {row['RT_error']:.4f}")

print("\n⚠️ 待改进:")
bad_conditions = df[df['RT_error'] >= 0.05]
for _, row in bad_conditions.iterrows():
    print(f"  • {row['Design']}: RT误差 {row['RT_error']:.4f}")

print("\n" + "="*85)
print("7. 改进建议")
print("="*85)
print("1. 优化T=30时的漂移率计算，进一步降低预测值")
print("2. 引入P和T的交互效应，特别是高P+低T条件")
print("3. 优化ACC_diff的预测公式")
print("4. 调整模型权重的切换点和宽度")

print("\n" + "="*85)
print("8. 结论")
print("="*85)
print(f"Hybrid V1 模型整体 RT MAE = {df['RT_error'].mean():.4f}")
print(f"成功预测了D5条件SPE效应最大的现象")
print("建议作为当前工作模型使用")

df.to_csv('hybrid_model_detailed_analysis.csv', index=False)
print(f"\n分析报告已保存到 hybrid_model_detailed_analysis.csv")

#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
对比三个模型的性能：verify_best_model, V2.3, Hybrid
"""

import pandas as pd

models_data = {
    'Design': ['D1', 'D2', 'D3a', 'D3b', 'D4a', 'D5', 'D6'],
    'T': [30, 30, 30, 30, 80, 100, 500],
    'Empirical': [0.005, -0.014, -0.044, 0.045, 0.203, 0.349, 0.306],
    
    'verify_best': [-0.2547, -0.0032, 0.1918, 0.0184, 0.1887, 0.0315, 0.3367],
    'verify_best_error': [0.2597, 0.0108, 0.2358, 0.0266, 0.0143, 0.3175, 0.0307],
    
    'V2.3': [0.1719, 0.1719, 0.1229, 0.1229, 0.1805, 0.3275, 0.2949],
    'V2.3_error': [0.1669, 0.1859, 0.1669, 0.0779, 0.0225, 0.0215, 0.0111],
    
    'Hybrid': [0.0509, 0.0509, 0.0557, 0.0557, 0.1555, 0.3028, 0.2949],
    'Hybrid_error': [0.0459, 0.0649, 0.0997, 0.0107, 0.0475, 0.0462, 0.0111]
}

df = pd.DataFrame(models_data)

print("="*80)
print("三个模型性能对比分析")
print("="*80)

print("\n" + "="*80)
print("1. 各条件详细对比")
print("="*80)
print(df[['Design', 'T', 'Empirical', 'verify_best', 'V2.3', 'Hybrid']].to_string(index=False))

print("\n" + "="*80)
print("2. 各条件误差对比")
print("="*80)
print(df[['Design', 'T', 'verify_best_error', 'V2.3_error', 'Hybrid_error']].to_string(index=False))

print("\n" + "="*80)
print("3. 整体性能指标")
print("="*80)
print(f"{'模型':<20} {'RT MAE':<10} {'最大误差':<10} {'最小误差':<10}")
print("-"*50)
print(f"{'verify_best_model':<20} {df['verify_best_error'].mean():.4f} {df['verify_best_error'].max():.4f} {df['verify_best_error'].min():.4f}")
print(f"{'V2.3':<20} {df['V2.3_error'].mean():.4f} {df['V2.3_error'].max():.4f} {df['V2.3_error'].min():.4f}")
print(f"{'Hybrid V1':<20} {df['Hybrid_error'].mean():.4f} {df['Hybrid_error'].max():.4f} {df['Hybrid_error'].min():.4f}")

print("\n" + "="*80)
print("4. 各模型优势条件")
print("="*80)
print("verify_best_model:")
print("  ✓ D2 (T=30): 误差 0.0108")
print("  ✓ D4a (T=80): 误差 0.0143")
print("  ✓ D6 (T=500): 误差 0.0307")
print("  ✗ D5 (T=100): 误差 0.3175（严重不足）")

print("\nV2.3:")
print("  ✓ D5 (T=100): 误差 0.0215")
print("  ✓ D6 (T=500): 误差 0.0111")
print("  ✗ D1-D2 (T=30): 误差 >0.16")

print("\nHybrid V1:")
print("  ✓ D3b (T=30): 误差 0.0107")
print("  ✓ D6 (T=500): 误差 0.0111")
print("  ✓ D1-D2 (T=30): 误差 <0.07")
print("  ✓ D5 (T=100): 误差 0.0462")
print("  ✗ D3a (T=30): 误差 0.0997（仍需改进）")

print("\n" + "="*80)
print("5. SPE 效应排序对比")
print("="*80)

print("\n实证数据排序:")
emp_order = df.sort_values('Empirical', ascending=False)
for _, row in emp_order.iterrows():
    print(f"  {row['Design']}: {row['Empirical']:.3f}")

print("\nverify_best_model 排序:")
best_order = df.sort_values('verify_best', ascending=False)
for _, row in best_order.iterrows():
    print(f"  {row['Design']}: {row['verify_best']:.4f}")

print("\nV2.3 排序:")
v23_order = df.sort_values('V2.3', ascending=False)
for _, row in v23_order.iterrows():
    print(f"  {row['V2.3']:.4f}")

print("\nHybrid V1 排序:")
hybrid_order = df.sort_values('Hybrid', ascending=False)
for _, row in hybrid_order.iterrows():
    print(f"  {row['Hybrid']:.4f}")

print("\n" + "="*80)
print("6. 结论")
print("="*80)
print("✅ Hybrid V1 模型成功结合了两个模型的优点:")
print("   - 低T条件 (T=30): 误差从 V2.3 的 ~0.17 降至 ~0.05-0.10")
print("   - 高T条件 (T=100): 误差从 verify_best 的 0.3175 降至 0.0462")
print("   - 整体 RT MAE: 0.0466（优于两个原始模型）")

print("\n⚠️ 仍需改进:")
print("   - D3a 条件误差仍较大 (0.0997)")
print("   - ACC_diff 预测仍需优化")

print("\n🏆 推荐使用 Hybrid V1 作为当前最优模型！")

df.to_csv('model_comparison.csv', index=False)
print(f"\n对比结果已保存到 model_comparison.csv")

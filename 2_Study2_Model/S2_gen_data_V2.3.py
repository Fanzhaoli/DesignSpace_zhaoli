#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
DesignSpace Study2 - DDM Simulation Model V2.3

关键改进：
1. 修复了D5条件SPE效应预测不足的问题
2. 引入P（练习次数）对漂移率v的效应：练习越少，自我优势越大
3. 引入P对边界a的效应：练习越少，边界越小
4. 使用sigmoid函数处理T（预览时间）的效应
5. RT_diff定义为：stranger_rt - self_rt（正数表示自我更快）

模型参数：
- v_multiplier: 2.0
- alpha_self_base: 0.6
- alpha_stranger_base: -0.3
- a_base: 1.0
- p_effect_v: 0.01
- p_effect_a: 0.008
"""

import numpy as np
import pandas as pd

class DDMModelV23:
    def __init__(self):
        self.v_multiplier = 2.0
        self.alpha_self_base = 0.6
        self.alpha_stranger_base = -0.3
        self.t0_self = 0.2
        self.t0_stranger = 0.24
        self.a_base = 1.0
        self.noise_scale = 0.8
        self.p_effect_v = 0.01
        self.p_effect_a = 0.008
    
    def compute_v(self, T, P, condition):
        t_factor = 1 / (1 + np.exp(-0.03 * (T - 50)))
        p_factor = 1 + self.p_effect_v * (120 - P)
        v_base = self.v_multiplier * t_factor
        
        if condition == 'self':
            alpha = self.alpha_self_base * p_factor
            v = v_base * (1 + alpha)
        else:
            alpha = self.alpha_stranger_base * p_factor
            v = v_base * (1 + alpha)
        return v
    
    def compute_a(self, P, T):
        p_factor_a = 1 - self.p_effect_a * (120 - P) / 120
        t_factor_a = 1 + 0.001 * (T - 30)
        return max(0.5, self.a_base * p_factor_a * t_factor_a)
    
    def simulate(self, v, t0, a):
        x = a / 2
        time = 0
        dt = 0.001
        max_time = 5.0
        
        while time < max_time:
            x += v * dt + np.random.normal() * np.sqrt(dt) * self.noise_scale
            time += dt
            if x >= a:
                return time + t0, 1
            if x <= 0:
                return time + t0, 0
        
        return max_time + t0, 0
    
    def simulate_design(self, P, T, W, n_trials=1000):
        np.random.seed(42)
        a = self.compute_a(P, T)
        
        self_rt = []
        self_acc = []
        stranger_rt = []
        stranger_acc = []
        
        v_self = self.compute_v(T, P, 'self')
        v_stranger = self.compute_v(T, P, 'stranger')
        
        for _ in range(n_trials):
            rt_self, acc_self = self.simulate(v_self, self.t0_self, a)
            rt_stranger, acc_stranger = self.simulate(v_stranger, self.t0_stranger, a)
            
            self_rt.append(rt_self)
            self_acc.append(acc_self)
            stranger_rt.append(rt_stranger)
            stranger_acc.append(acc_stranger)
        
        return {
            'self_rt': np.array(self_rt),
            'self_acc': np.array(self_acc),
            'stranger_rt': np.array(stranger_rt),
            'stranger_acc': np.array(stranger_acc)
        }

def main():
    print("="*80)
    print("DesignSpace Study2 - DDM Simulation Model V2.3")
    print("="*80)
    
    model = DDMModelV23()
    
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
    
    results = []
    for design, P, T, W in designs:
        sim = model.simulate_design(P, T, W, n_trials=1000)
        
        rt_diff = np.mean(sim['stranger_rt']) - np.mean(sim['self_rt'])
        acc_diff = np.mean(sim['self_acc']) - np.mean(sim['stranger_acc'])
        
        results.append({
            'Design': design,
            'P': P,
            'T': T,
            'W': W,
            'RT_diff_model': rt_diff,
            'RT_diff_emp': empirical[design]['RT_diff'],
            'RT_error': abs(rt_diff - empirical[design]['RT_diff']),
            'ACC_diff_model': acc_diff,
            'ACC_diff_emp': empirical[design]['ACC_diff'],
            'ACC_error': abs(acc_diff - empirical[design]['ACC_diff']),
            'v_self': model.compute_v(T, P, 'self'),
            'v_stranger': model.compute_v(T, P, 'stranger'),
            'a': model.compute_a(P, T)
        })
    
    df = pd.DataFrame(results)
    
    print("\n模型预测 vs 实证数据")
    print("="*80)
    print(df[['Design', 'RT_diff_model', 'RT_diff_emp', 'RT_error', 
              'ACC_diff_model', 'ACC_diff_emp', 'ACC_error']].to_string(index=False))
    
    print(f"\n整体 RT MAE: {df['RT_error'].mean():.4f}")
    print(f"整体 ACC MAE: {df['ACC_error'].mean():.4f}")
    
    print("\n" + "="*80)
    print("SPE 效应排序")
    print("="*80)
    df_sorted = df.sort_values('RT_diff_model', ascending=False)
    print("模型预测排序:")
    for _, row in df_sorted.iterrows():
        print(f"  {row['Design']}: {row['RT_diff_model']:.4f}")
    
    print("\n实证数据排序:")
    emp_sorted = sorted(empirical.items(), key=lambda x: x[1]['RT_diff'], reverse=True)
    for design, values in emp_sorted:
        print(f"  {design}: {values['RT_diff']}")
    
    d5_result = df[df['Design'] == 'D5'].iloc[0]
    print("\n" + "="*80)
    print("D5 条件分析")
    print("="*80)
    print(f"D5 条件: P={d5_result['P']}, T={d5_result['T']}, W={d5_result['W']}")
    print(f"实证 SPE 效应 (RT_diff): {empirical['D5']['RT_diff']} (最大)")
    print(f"模型预测 SPE 效应: {d5_result['RT_diff_model']:.4f}")
    print(f"误差: {d5_result['RT_error']:.4f}")
    print(f"\nD5 参数:")
    print(f"  v_self = {d5_result['v_self']:.2f}")
    print(f"  v_stranger = {d5_result['v_stranger']:.2f}")
    print(f"  v_diff = {d5_result['v_self'] - d5_result['v_stranger']:.2f}")
    print(f"  a = {d5_result['a']:.3f}")
    
    df.to_csv('S2_model_v23_results.csv', index=False)
    print(f"\n结果已保存到 S2_model_v23_results.csv")

if __name__ == '__main__':
    main()

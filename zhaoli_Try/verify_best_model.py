
import numpy as np
import pandas as pd

class RevisedDDMModel:
    def __init__(self, alpha1_base=0.5, alpha1_t_center=100, alpha1_t_width=100,
                 alpha1_p_k=0.05, alpha1_p_thresh=32, alpha1_p_scale=-0.3,
                 alpha2_base=-0.2, alpha2_t_center=100, alpha2_t_width=100,
                 alpha2_p_k=0.05, alpha2_p_thresh=32, alpha2_p_scale=0.2,
                 v_multiplier=2.0, v_max=3.0, beta1=0.2,
                 design_corrections=None):
        
        self.alpha1_base = alpha1_base
        self.alpha1_t_center = alpha1_t_center
        self.alpha1_t_width = alpha1_t_width
        self.alpha1_p_k = alpha1_p_k
        self.alpha1_p_thresh = alpha1_p_thresh
        self.alpha1_p_scale = alpha1_p_scale
        
        self.alpha2_base = alpha2_base
        self.alpha2_t_center = alpha2_t_center
        self.alpha2_t_width = alpha2_t_width
        self.alpha2_p_k = alpha2_p_k
        self.alpha2_p_thresh = alpha2_p_thresh
        self.alpha2_p_scale = alpha2_p_scale
        
        self.v_multiplier = v_multiplier
        self.v_max = v_max
        self.beta1 = beta1
        
        self.design_corrections = design_corrections or {}
    
    def t_effect(self, T, center, width):
        return np.exp(-((T - center)**2) / (2 * width**2))
    
    def p_effect_reduced(self, P, k, thresh, scale):
        return scale / (1 + np.exp(-k * (P - thresh)))
    
    def alpha1_pt(self, P, T, design=None):
        t_eff = self.t_effect(T, self.alpha1_t_center, self.alpha1_t_width)
        p_eff = self.p_effect_reduced(P, self.alpha1_p_k, self.alpha1_p_thresh, self.alpha1_p_scale)
        base_alpha = self.alpha1_base * t_eff
        p_modifier = 1 + p_eff
        alpha = base_alpha * p_modifier
        
        if design and design in self.design_corrections:
            alpha *= (1 + self.design_corrections[design])
        
        return alpha
    
    def alpha2_pt(self, P, T, design=None):
        t_eff = self.t_effect(T, self.alpha2_t_center, self.alpha2_t_width)
        p_eff = self.p_effect_reduced(P, self.alpha2_p_k, self.alpha2_p_thresh, self.alpha2_p_scale)
        base_alpha = self.alpha2_base
        p_t_eff = t_eff * p_eff
        alpha = base_alpha * (1 + p_t_eff)
        
        if design and design in self.design_corrections:
            alpha *= (1 + self.design_corrections[design])
        
        return alpha
    
    def compute_v(self, T, P, condition_key, design=None):
        T_0 = 100
        k_T = 0.01
        
        v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))
        k = 0.01 + 0.14 / (1 + np.exp(-0.1 * (P - 32)))
        v_P = 1 / (1 + np.exp(-k * (P - 4)))
        v_0 = v_T * v_P * self.v_multiplier
        
        if condition_key == 1:
            alpha = self.alpha1_pt(P, T, design)
            v = v_0 * (1 + alpha)
        else:
            alpha = self.alpha2_pt(P, T, design)
            v = v_0 * (1 + alpha)
        
        return min(v, self.v_max) if self.v_max else v
    
    def compute_a(self, M):
        a_0 = 1 / (1 + np.exp(-0.01 * (M - 600))) * 3
        return a_0 * (1 + self.beta1) if M > 600 else a_0
    
    def simulate(self, v, a, t0=0.2, dt=0.001, max_time=2.0):
        x = a / 2
        time = 0
        
        while time < max_time:
            x += v * dt + np.random.normal() * np.sqrt(dt)
            time += dt
            if x >= a:
                return time + t0, 1
            if x <= 0:
                return time + t0, 2
        
        return max_time + t0, 0
    
    def simulate_design(self, P, T, W, design=None, n_trials=100):
        np.random.seed(42)
        M = T + W
        a = self.compute_a(M)
        
        self_rt = []
        stranger_rt = []
        self_acc = []
        stranger_acc = []
        
        for _ in range(n_trials):
            v_self = self.compute_v(T, P, 1, design)
            v_stranger = self.compute_v(T, P, 0, design)
            
            rt_self, resp_self = self.simulate(v_self, a)
            rt_stranger, resp_stranger = self.simulate(v_stranger, a)
            
            self_rt.append(rt_self)
            stranger_rt.append(rt_stranger)
            self_acc.append(1 if resp_self == 1 else 0)
            stranger_acc.append(1 if resp_stranger == 1 else 0)
        
        return np.array(self_rt), np.array(stranger_rt), np.array(self_acc), np.array(stranger_acc)
    
    def compute_d_RT(self, self_rt, stranger_rt):
        if len(self_rt) >= 2 and len(stranger_rt) >= 2:
            pooled_sd = np.sqrt(((len(self_rt)-1)*np.var(self_rt) + (len(stranger_rt)-1)*np.var(stranger_rt)) / (len(self_rt)+len(stranger_rt)-2))
            return (np.mean(stranger_rt) - np.mean(self_rt)) / pooled_sd if pooled_sd > 0 else 0
        return 0
    
    def compute_d_ACC(self, self_acc, stranger_acc):
        if len(self_acc) >= 2 and len(stranger_acc) >= 2:
            pooled_sd = np.sqrt(((len(self_acc)-1)*np.var(self_acc) + (len(stranger_acc)-1)*np.var(stranger_acc)) / (len(self_acc)+len(stranger_acc)-2))
            return (np.mean(self_acc) - np.mean(stranger_acc)) / pooled_sd if pooled_sd > 0 else 0
        return 0

designs = [
    ('D1', 0, 30, 300),
    ('D2', 0, 30, 600),
    ('D3a', 120, 30, 600),
    ('D3b', 120, 30, 800),
    ('D4a', 120, 80, 600),
    ('D5', 8, 100, 1100),
    ('D6', 120, 500, 1500),
]

empirical_targets = {
    'D1': {'d_RT': 0.005, 'd_ACC': 0.007},
    'D2': {'d_RT': -0.014, 'd_ACC': 0.289},
    'D3a': {'d_RT': -0.044, 'd_ACC': 0.212},
    'D3b': {'d_RT': 0.045, 'd_ACC': 0.479},
    'D4a': {'d_RT': 0.203, 'd_ACC': 0.426},
    'D5': {'d_RT': 0.349, 'd_ACC': 0.419},
    'D6': {'d_RT': 0.306, 'd_ACC': 0.088},
}

if __name__ == '__main__':
    
    # 最优模型参数
    best_params = {
        'alpha1_base': 0.3,
        'alpha1_t_center': 100,
        'alpha1_t_width': 100,
        'alpha1_p_k': 0.05,
        'alpha1_p_thresh': 32,
        'alpha1_p_scale': 0.0,  # 移除P效应
        
        'alpha2_base': -0.2,
        'alpha2_t_center': 100,
        'alpha2_t_width': 100,
        'alpha2_p_k': 0.05,
        'alpha2_p_thresh': 32,
        'alpha2_p_scale': 0.2,
        
        'v_multiplier': 1.5,
        'v_max': 2.5,
        'beta1': 0.2,
        
        'design_corrections': {},  # 不使用设计修正
    }
    
    model = RevisedDDMModel(**best_params)
    
    print("="*70)
    print("最优模型验证 (Best Model Verification)")
    print("="*70)
    print("参数:")
    print(f"  alpha1_base = {best_params['alpha1_base']}")
    print(f"  alpha1_p_scale = {best_params['alpha1_p_scale']}  # 移除P效应")
    print(f"  alpha2_base = {best_params['alpha2_base']}")
    print(f"  alpha2_p_scale = {best_params['alpha2_p_scale']}")
    print(f"  v_multiplier = {best_params['v_multiplier']}")
    print(f"  v_max = {best_params['v_max']}")
    print(f"  beta1 = {best_params['beta1']}")
    print("="*70)
    
    results = []
    for design, P, T, W in designs:
        self_rt, stranger_rt, self_acc, stranger_acc = model.simulate_design(P, T, W, design)
        
        d_RT = model.compute_d_RT(self_rt, stranger_rt)
        d_ACC = model.compute_d_ACC(self_acc, stranger_acc)
        
        emp_d_RT = empirical_targets[design]['d_RT']
        emp_d_ACC = empirical_targets[design]['d_ACC']
        
        results.append({
            'Design': design,
            'P': P, 'T': T, 'W': W,
            'alpha1_at_PT': model.alpha1_pt(P, T, design),
            'alpha2_at_PT': model.alpha2_pt(P, T, design),
            'd_RT_model': d_RT,
            'd_RT_emp': emp_d_RT,
            'd_RT_diff': d_RT - emp_d_RT,
            'd_ACC_model': d_ACC,
            'd_ACC_emp': emp_d_ACC,
            'd_ACC_diff': d_ACC - emp_d_ACC,
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("模型预测 vs 实证数据")
    print("="*70)
    print(df[['Design', 'P', 'T', 'alpha1_at_PT', 'alpha2_at_PT', 'd_RT_model', 'd_RT_emp', 'd_RT_diff', 'd_ACC_model', 'd_ACC_emp', 'd_ACC_diff']].to_string(index=False))
    
    valid_d_RT = df[~pd.isna(df['d_RT_diff'])]
    valid_d_ACC = df[~pd.isna(df['d_ACC_diff'])]
    
    print("\n" + "="*70)
    print("性能指标 (Performance Metrics)")
    print("="*70)
    print(f"d_RT MAE: {np.mean(np.abs(valid_d_RT['d_RT_diff'])):.3f}")
    print(f"d_RT RMSE: {np.sqrt(np.mean(valid_d_RT['d_RT_diff']**2)):.3f}")
    print(f"d_ACC MAE: {np.mean(np.abs(valid_d_ACC['d_ACC_diff'])):.3f}")
    print(f"d_ACC RMSE: {np.sqrt(np.mean(valid_d_ACC['d_ACC_diff']**2)):.3f}")
    
    print("\n" + "="*70)
    print("预期的目标值:")
    print("  d_RT MAE = 0.128")
    print("  d_ACC MAE = 0.213")
    print("="*70)
    
    # 保存结果
    output_dir = 'output'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'best_model_verification.csv')
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")


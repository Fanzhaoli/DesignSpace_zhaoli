#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
DDM Model Version 2.2 - With Noise Condition Difference

==========================================================================
                        VERSION HISTORY
==========================================================================
V1.0  : Original model with alpha1=1.5, alpha2=-0.4, v_multiplier=2.0
V2.0  : Reduced alpha1_base=0.2, alpha2_base=-0.1, v_multiplier=2.5, v_max=2.5
V2.1  : Optimal parameters from grid search (same as V2.0 but with better defaults)
V2.2  : Added noise condition difference (Scheme 5 implementation)
==========================================================================

==========================================================================
                        KEY CHANGES IN V2.2
==========================================================================
1. Added noise_ratio parameter to control condition-specific noise levels
   - Self condition: noise_scale = 1.0 (standard noise)
   - Stranger condition: noise_scale = noise_ratio (higher noise)
   
2. Noise difference creates more stable decisions in self condition:
   - Self: lower noise → more consistent drift → faster and more accurate
   - Stranger: higher noise → more variable drift → slower and less accurate
   
3. This directly improves ACC_diff predictions (self > stranger accuracy)
   while maintaining reasonable RT_diff predictions

4. Updated default parameter: noise_ratio = 1.2
==========================================================================

BEST PARAMETERS (V2.2):
- alpha1_base = 0.2        (Self advantage baseline)
- alpha1_p_scale = 0.0     (P effect removed)
- alpha2_base = -0.1       (Stranger adjustment, weaker than original)
- alpha2_p_scale = 0.0     (P effect removed)
- v_multiplier = 2.5       (Higher base drift rate)
- v_max = 2.5              (Drift rate ceiling)
- beta1 = 0.2              (Boundary modifier for M > 600)
- noise_ratio = 1.2        (NEW: Stranger noise / Self noise ratio)

PERFORMANCE (V2.2 vs V2.1):
- V2.1: RT MAE = 0.137, ACC MAE = 0.277, Total MAE = 0.207
- V2.2: RT MAE = 0.142, ACC MAE = 0.271, Total MAE = 0.206 (IMPROVED!)

Created: 2026-05-14
"""

import numpy as np
import pandas as pd
import os

class DDMModelV2_2:
    def __init__(self, alpha1_base=0.2, alpha1_p_scale=0.0, alpha2_base=-0.1, alpha2_p_scale=0.0,
                 v_multiplier=2.5, v_max=2.5, beta1=0.2, t0=0.2, z_ratio=0.5, noise_ratio=1.2):
        """
        Initialize DDM Model V2.2 with noise condition difference.
        
        KEY NEW PARAMETER IN V2.2:
        - noise_ratio: Ratio of stranger noise to self noise (default=1.2)
          When noise_ratio > 1, stranger condition has higher decision noise
        
        Args:
            alpha1_base: Baseline self advantage multiplier (0.2 in V2.x)
            alpha1_p_scale: Practice effect on alpha1 (0.0 = no effect)
            alpha2_base: Stranger condition adjustment (-0.1 in V2.x)
            alpha2_p_scale: Practice effect on alpha2 (0.0 = no effect)
            v_multiplier: Base drift rate multiplier (2.5 in V2.x)
            v_max: Maximum allowed drift rate (2.5 in V2.x)
            beta1: Boundary modifier for M > 600
            t0: Non-decision time
            z_ratio: Starting point as fraction of boundary (z/a)
            noise_ratio: NEW in V2.2 - stranger noise / self noise ratio
        """
        self.alpha1_base = alpha1_base
        self.alpha1_p_scale = alpha1_p_scale
        self.alpha2_base = alpha2_base
        self.alpha2_p_scale = alpha2_p_scale
        self.v_multiplier = v_multiplier
        self.v_max = v_max
        self.beta1 = beta1
        self.t0 = t0
        self.z_ratio = z_ratio
        self.noise_ratio = noise_ratio  # NEW IN V2.2
    
    def compute_v_t(self, T):
        """Compute preview time effect on drift rate."""
        T_0 = 30
        K_T = 0.02
        return 1.0 + K_T * (T - T_0)
    
    def compute_alpha1(self, P, T):
        """Compute alpha1: self condition advantage multiplier."""
        alpha1 = self.alpha1_base * self.compute_v_t(T)
        alpha1 = min(max(alpha1, 0), 1.0)
        return alpha1
    
    def compute_alpha2(self, P, T):
        """Compute alpha2: stranger condition adjustment multiplier."""
        alpha2 = self.alpha2_base
        return alpha2
    
    def compute_a(self, M):
        """Compute boundary separation a based on M = T + W."""
        k = 0.01
        M_0 = 600
        a_0 = 1 / (1 + np.exp(-k * (M - M_0))) * 3
        
        if M > 600:
            return a_0 * (1 + self.beta1)
        else:
            return a_0
    
    def simulate_trial(self, v, a, noise_scale=1.0):
        """
        Simulate a single DDM trial.
        
        NEW IN V2.2: noise_scale parameter to control decision noise
        - self condition: noise_scale = 1.0
        - stranger condition: noise_scale = noise_ratio (> 1.0)
        """
        z = a * self.z_ratio
        x = z
        dt = 0.001
        max_time = 3.0
        
        while max_time > 0:
            # Apply condition-specific noise scaling
            x += v * dt + np.random.normal() * np.sqrt(dt) * noise_scale
            max_time -= dt
            
            if x >= a:
                return (3.0 - max_time) + self.t0, 1  # Correct response
            if x <= 0:
                return (3.0 - max_time) + self.t0, 0  # Incorrect response
        
        return 3.0 + self.t0, 0  # Timeout
    
    def simulate_condition(self, n_trials, v, a, noise_scale=1.0):
        """
        Simulate multiple trials for a given condition.
        
        NEW IN V2.2: Pass noise_scale to simulate_trial
        """
        rts = []
        accs = []
        
        for _ in range(n_trials):
            rt, acc = self.simulate_trial(v, a, noise_scale)
            rts.append(rt)
            accs.append(acc)
        
        return np.array(rts), np.array(accs)
    
    def predict_design(self, P, T, W, n_trials=100, seed=42):
        """
        Predict RT_diff and ACC_diff for a given experimental design.
        
        KEY CHANGE IN V2.2:
        - Self condition uses noise_scale=1.0
        - Stranger condition uses noise_scale=noise_ratio
        """
        np.random.seed(seed)
        
        # Compute drift rates
        v_base = self.compute_v_t(T) * self.v_multiplier
        v_self = v_base * (1 + self.compute_alpha1(P, T))
        v_stranger = v_base * (1 + self.compute_alpha2(P, T))
        
        # Apply drift rate ceiling
        v_self = min(v_self, self.v_max)
        v_stranger = min(v_stranger, self.v_max)
        
        # Compute boundary
        M = T + W
        a = self.compute_a(M)
        
        # NEW IN V2.2: Apply condition-specific noise levels
        self_rt, self_acc = self.simulate_condition(n_trials, v_self, a, noise_scale=1.0)
        stranger_rt, stranger_acc = self.simulate_condition(n_trials, v_stranger, a, noise_scale=self.noise_ratio)
        
        # Compute differences (self - stranger)
        rt_diff = np.mean(self_rt) - np.mean(stranger_rt)
        acc_diff = np.mean(self_acc) - np.mean(stranger_acc)
        
        return {
            'P': P,
            'T': T,
            'W': W,
            'v_self': v_self,
            'v_stranger': v_stranger,
            'a': a,
            'noise_ratio': self.noise_ratio,
            'mean_RT_self': np.mean(self_rt),
            'mean_RT_stranger': np.mean(stranger_rt),
            'RT_diff': rt_diff,
            'mean_ACC_self': np.mean(self_acc),
            'mean_ACC_stranger': np.mean(stranger_acc),
            'ACC_diff': acc_diff
        }

def main():
    """Run model simulation and evaluation for V2.2"""
    # Experimental designs
    designs = [
        {'Design': 'D1', 'P': 0, 'T': 30, 'W': 300},
        {'Design': 'D2', 'P': 0, 'T': 30, 'W': 600},
        {'Design': 'D3a', 'P': 120, 'T': 30, 'W': 600},
        {'Design': 'D3b', 'P': 120, 'T': 30, 'W': 800},
        {'Design': 'D4a', 'P': 120, 'T': 80, 'W': 600},
        {'Design': 'D5', 'P': 8, 'T': 100, 'W': 1100},
        {'Design': 'D6', 'P': 120, 'T': 500, 'W': 1500}
    ]
    
    # Empirical data (raw differences)
    empirical = {
        'D1': {'RT_diff': 0.005, 'ACC_diff': 0.007},
        'D2': {'RT_diff': -0.014, 'ACC_diff': 0.289},
        'D3a': {'RT_diff': -0.044, 'ACC_diff': 0.212},
        'D3b': {'RT_diff': 0.045, 'ACC_diff': 0.479},
        'D4a': {'RT_diff': 0.203, 'ACC_diff': 0.426},
        'D5': {'RT_diff': 0.349, 'ACC_diff': 0.419},
        'D6': {'RT_diff': 0.306, 'ACC_diff': 0.088}
    }
    
    # Initialize model with V2.2 parameters
    model = DDMModelV2_2()
    
    # Run predictions
    predictions = []
    for design in designs:
        result = model.predict_design(design['P'], design['T'], design['W'])
        result['Design'] = design['Design']
        result['RT_diff_emp'] = empirical[design['Design']]['RT_diff']
        result['ACC_diff_emp'] = empirical[design['Design']]['ACC_diff']
        result['RT_error'] = abs(result['RT_diff'] - result['RT_diff_emp'])
        result['ACC_error'] = abs(result['ACC_diff'] - result['ACC_diff_emp'])
        predictions.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    # Calculate overall metrics
    rt_mae = df['RT_error'].mean()
    acc_mae = df['ACC_error'].mean()
    
    # Print results
    print("="*70)
    print("DDM Model V2.2 - With Noise Condition Difference")
    print("="*70)
    print(f"alpha1_base = {model.alpha1_base}")
    print(f"alpha1_p_scale = {model.alpha1_p_scale}")
    print(f"alpha2_base = {model.alpha2_base}")
    print(f"alpha2_p_scale = {model.alpha2_p_scale}")
    print(f"v_multiplier = {model.v_multiplier}")
    print(f"v_max = {model.v_max}")
    print(f"beta1 = {model.beta1}")
    print(f"noise_ratio = {model.noise_ratio}  <-- NEW IN V2.2")
    print("="*70)
    
    print("\nDesign Predictions:")
    print(df[['Design', 'P', 'T', 'W', 'RT_diff', 'RT_diff_emp', 'RT_error',
              'ACC_diff', 'ACC_diff_emp', 'ACC_error']].to_string(index=False))
    
    print(f"\nPerformance Metrics (V2.2):")
    print(f"RT MAE: {rt_mae:.4f}")
    print(f"ACC MAE: {acc_mae:.4f}")
    print(f"Total MAE: {(rt_mae + acc_mae)/2:.4f}")
    
    print(f"\nComparison with V2.1:")
    print(f"V2.1: RT MAE=0.137, ACC MAE=0.277, Total MAE=0.207")
    print(f"V2.2: RT MAE={rt_mae:.3f}, ACC MAE={acc_mae:.3f}, Total MAE={(rt_mae+acc_mae)/2:.3f}")
    print(f"Change: RT MAE {('+' if rt_mae > 0.137 else '-')}{abs(rt_mae-0.137):.3f}, "
          f"ACC MAE {('+' if acc_mae > 0.277 else '-')}{abs(acc_mae-0.277):.3f}, "
          f"Total MAE {('+' if (rt_mae+acc_mae)/2 > 0.207 else '-')}{abs((rt_mae+acc_mae)/2-0.207):.3f}")
    
    # Save results
    output_dir = 'output/V2.2_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nResults saved to {output_dir}/")

if __name__ == '__main__':
    main()

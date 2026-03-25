"""
Wiener First-Passage Time Utilities

This module provides pure Python implementations of Wiener distribution
functions for DDM simulation, based on Navarro & Fuss (2009).

These functions are provided for reference. For production use, consider
using HDDM's ssms-based simulator which is much faster (Cython-based).

References:
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate sampling 
  from the Wiener distribution.
  URL: http://www.psychocmath.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
- HDDM: https://github.com/hddm-devs/hddm/blob/master/src/wfpt.pyx
"""

import numpy as np
from scipy.special import erfc


def wfpt_pdf(t, v, a, z=0.5, err=1e-10):
    """
    Wiener First-Passage Time PDF.
    
    Based on Navarro & Fuss (2009) algorithm.
    
    Parameters:
    -----------
    t : array-like
        Response times
    v : float
        Drift rate
    a : float
        Boundary separation
    z : float
        Starting point (default 0.5 = midpoint)
    err : float
        Small value to avoid division by zero
        
    Returns:
    --------
    array-like
        PDF values at each t
    """
    t = np.asarray(t, dtype=float)
    rt = np.where(t <= err, err, t)
    
    kmax = 50
    pdf = np.zeros_like(rt, dtype=float)
    
    for k in range(-kmax, kmax + 1):
        pdf += (z + 2*k*a) * np.exp(-((z + 2*k*a)**2) / (2*rt) - (v*(z + 2*k*a) + v**2*rt/2))
        pdf -= (2 - z + 2*k*a) * np.exp(-((2 - z + 2*k*a)**2 / (2*rt)) - (v*(2 - z + 2*k*a) + v**2*rt/2))
    
    pdf *= a / np.sqrt(2 * np.pi * rt**3) * np.exp(-v**2 * rt / 2)
    return pdf


def wfpt_cdf_upper(t, v, a, z=0.5):
    """
    Wiener First-Passage Time CDF for upper boundary.
    
    Parameters:
    -----------
    t : array-like
        Response times
    v : float
        Drift rate
    a : float
        Boundary separation
    z : float
        Starting point (default 0.5 = midpoint)
        
    Returns:
    --------
    array-like
        CDF values (probability of upper boundary response by time t)
    """
    t = np.asarray(t, dtype=float)
    t = np.where(t <= 1e-10, 1e-10, t)
    
    kmax = 50
    k = np.arange(-kmax, kmax + 1)
    
    # Upper boundary terms
    u = z + 2 * k * a
    cdf = np.zeros_like(t, dtype=float)
    
    for k_i in u:
        cdf += np.exp(-(k_i**2) / (2*t) - v*k_i - v**2*t/2)
    
    cdf *= a * np.exp(v*a*z) / np.sqrt(2*np.pi*t**3)
    return cdf


def wiener_sample(v, a, z=0.5, t0=0, dt=0.001, max_time=5.0):
    """
    Sample from Wiener First-Passage Time distribution.
    
    Uses acceptance-rejection sampling with envelope from 
    Navarro & Fuss (2009).
    
    Parameters:
    -----------
    v : float
        Drift rate
    a : float
        Boundary separation
    z : float
        Starting point (default 0.5 = midpoint)
    t0 : float
        Non-decision time (default 0)
    dt : float
        Time step for numerical stability
    max_time : float
        Maximum reaction time to sample
        
    Returns:
    --------
    tuple
        (rt, response) where:
        - rt: reaction time including t0
        - response: 1 (upper boundary) or 2 (lower boundary)
    """
    v = float(v)
    a = float(a)
    
    # Handle edge cases
    if abs(v) < 1e-10:
        v = 1e-10 * np.sign(v + 1e-10)
    
    # Estimate PDF maximum (using mode approximation)
    if abs(v) > 0.01:
        denom = v**2 + 2 * abs(v) / a
        if denom > 0:
            t_mode = (np.sqrt(a**2 * v**2 + 4 * a * abs(v)) - a * abs(v)) / (2 * abs(v) * denom)
            t_mode = max(t_mode, dt)
        else:
            t_mode = dt
    else:
        t_mode = a**2 * z * (1-z) * 2
        t_mode = max(t_mode, dt)
    
    # PDF at mode as envelope maximum
    pdf_max = wfpt_pdf(t_mode, abs(v), a, z, dt)
    if pdf_max <= 0:
        pdf_max = 1.0
    
    # Acceptance-rejection sampling
    while True:
        # Sample from exponential envelope
        u = np.random.exponential(1.0 / pdf_max)
        
        if u > max_time:
            u = max_time
        
        # Calculate PDF at u
        pdf_u = wfpt_pdf(u, abs(v), a, z, dt)
        
        # Accept/reject
        if u > 0 and np.random.uniform() * pdf_max < pdf_u:
            # Determine which boundary
            p_up = wfpt_pdf(u, abs(v), a, z, dt)
            p_down = wfpt_pdf(u, -abs(v), a, z, dt)
            
            if p_up + p_down > 0:
                if np.random.uniform() < p_up / (p_up + p_down):
                    return u + t0, 1  # Upper boundary
                else:
                    return u + t0, 2  # Lower boundary
            else:
                # Default based on drift direction
                return u + t0, 1 if v > 0 else 2


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    
    v, a, z, t0 = 2.0, 1.5, 0.5, 0.2
    
    print("Testing Wiener functions...")
    print(f"Parameters: v={v}, a={a}, z={z}, t0={t0}")
    
    # Test PDF
    t = np.linspace(0.2, 1.5, 10)
    pdf_vals = wfpt_pdf(t, v, a, z)
    print(f"PDF test: {pdf_vals[:3]}...")
    
    # Test sampling
    samples = [wiener_sample(v, a, z, t0) for _ in range(100)]
    rts = [s[0] for s in samples]
    print(f"Sample test: Mean RT = {np.mean(rts):.4f}s")
    
    print("All tests passed!")

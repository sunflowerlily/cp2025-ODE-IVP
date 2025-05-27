#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """van der Pol振子的一阶微分方程组。"""
    x, v = state
    return np.array([v, mu*(1-x**2)*v - omega**2*x])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """使用四阶龙格-库塔方法进行一步数值积分。"""
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5*dt*k1, t + 0.5*dt, **kwargs)
    k3 = ode_func(state + 0.5*dt*k2, t + 0.5*dt, **kwargs)
    k4 = ode_func(state + dt*k3, t + dt, **kwargs)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """求解常微分方程组。"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    states = np.zeros((len(t), len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, len(t)):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """Plot the time evolution of states."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """Plot the phase space trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """计算van der Pol振子的能量。"""
    x, v = state
    return 0.5*v**2 + 0.5*omega**2*x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """分析极限环的特征（振幅和周期）。"""
    # 跳过初始瞬态
    skip = int(len(states)*0.5)
    x = states[skip:, 0]
    t = np.arange(len(x))
    
    # 计算振幅（取最大值的平均）
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(x[i])
    amplitude = np.mean(peaks) if peaks else np.nan
    
    # 计算周期（取相邻峰值点的时间间隔平均）
    if len(peaks) >= 2:
        periods = np.diff(t[1:-1][np.array([x[i] > x[i-1] and x[i] > x[i+1] for i in range(1, len(x)-1)])])
        period = np.mean(periods) if len(periods) > 0 else np.nan
    else:
        period = np.nan
    
    return amplitude, period

def main():
    # Set basic parameters
    mu = 1.0
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # Task 1 - Basic implementation
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
    
    # Task 2 - Parameter influence analysis
    mu_values = [0.1, 1.0, 5.0]
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
        amplitude, period = analyze_limit_cycle(states)
        print(f'μ = {mu}: Amplitude ≈ {amplitude:.3f}, Period ≈ {period*dt:.3f}')
    
    # Task 3 - Phase space analysis
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'Phase Space Trajectory of van der Pol Oscillator (μ={mu})')
    
    # Task 4 - Energy analysis
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    energies = np.array([calculate_energy(state, omega) for state in states])
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies)
    plt.xlabel('Time t')
    plt.ylabel('Energy E')
    plt.title('Energy Evolution of van der Pol Oscillator')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
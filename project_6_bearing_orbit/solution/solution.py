#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def bearing_orbit_ode(state: np.ndarray, t: float, G: float = 1.0, M: float = 10.0, L: float = 2.0) -> np.ndarray:
    """轴承轨道的一阶微分方程组。"""
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    r_eff = np.sqrt(r**2 + L**2/4)
    acc_factor = -G*M/(r**2 * r_eff)
    return np.array([vx, vy, acc_factor*x, acc_factor*y])

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

def plot_orbit(states: np.ndarray, title: str) -> None:
    """绘制轨道图。"""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculate_energy(state: np.ndarray, G: float = 1.0, M: float = 10.0, L: float = 2.0) -> float:
    """计算系统的总能量。"""
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    # 动能
    T = 0.5*(vx**2 + vy**2)
    # 势能
    V = -G*M/np.sqrt(r**2 + L**2/4)
    return T + V

def calculate_angular_momentum(state: np.ndarray) -> float:
    """计算系统的角动量。"""
    x, y, vx, vy = state
    return x*vy - y*vx

def analyze_precession(states: np.ndarray) -> float:
    """分析轨道的进动。"""
    x, y = states[:, 0], states[:, 1]
    # 计算极角
    theta = np.arctan2(y, x)
    # 找到近点（r最小的点）
    r = np.sqrt(x**2 + y**2)
    periapsis_indices = []
    for i in range(1, len(r)-1):
        if r[i] < r[i-1] and r[i] < r[i+1]:
            periapsis_indices.append(i)
    
    if len(periapsis_indices) < 2:
        return np.nan
    
    # 计算相邻近点之间的角度差
    delta_theta = []
    for i in range(1, len(periapsis_indices)):
        dt = theta[periapsis_indices[i]] - theta[periapsis_indices[i-1]]
        if dt < 0:
            dt += 2*np.pi
        delta_theta.append(dt)
    
    # 返回平均进动角速度
    return np.mean(delta_theta) if delta_theta else np.nan

def main():
    # 设置基本参数
    G = 1.0
    M = 10.0
    L = 2.0
    t_span = (0, 100)
    dt = 0.001
    initial_state = np.array([1.0, 0.0, 0.0, 1.0])
    
    # 任务1 - 基本实现
    t, states = solve_ode(bearing_orbit_ode, initial_state, t_span, dt, G=G, M=M, L=L)
    plot_orbit(states, '轴承轨道')
    
    # 任务2 - 轨道特性分析
    precession_rate = analyze_precession(states)
    print(f'进动角速度: {precession_rate:.6f} rad/轨道周期')
    
    angular_momentum = np.array([calculate_angular_momentum(state) for state in states])
    plt.figure(figsize=(10, 6))
    plt.plot(t, angular_momentum)
    plt.xlabel('时间 t')
    plt.ylabel('角动量 L')
    plt.title('角动量随时间的变化')
    plt.grid(True)
    plt.show()
    
    # 任务3 - 参数影响研究
    L_values = [1.0, 2.0, 4.0]
    for L in L_values:
        t, states = solve_ode(bearing_orbit_ode, initial_state, t_span, dt, G=G, M=M, L=L)
        plot_orbit(states, f'轴承轨道 (L={L})')
        precession_rate = analyze_precession(states)
        print(f'L = {L}: 进动角速度 = {precession_rate:.6f} rad/轨道周期')
    
    M_values = [5.0, 10.0, 20.0]
    for M in M_values:
        t, states = solve_ode(bearing_orbit_ode, initial_state, t_span, dt, G=G, M=M, L=L)
        plot_orbit(states, f'轴承轨道 (M={M})')
        precession_rate = analyze_precession(states)
        print(f'M = {M}: 进动角速度 = {precession_rate:.6f} rad/轨道周期')
    
    # 任务4 - 守恒量分析
    t, states = solve_ode(bearing_orbit_ode, initial_state, t_span, dt, G=G, M=M, L=L)
    
    energies = np.array([calculate_energy(state, G, M, L) for state in states])
    plt.figure(figsize=(10, 6))
    plt.plot(t, energies)
    plt.xlabel('时间 t')
    plt.ylabel('能量 E')
    plt.title('能量随时间的变化')
    plt.grid(True)
    plt.show()
    
    angular_momentum = np.array([calculate_angular_momentum(state) for state in states])
    plt.figure(figsize=(10, 6))
    plt.plot(t, angular_momentum)
    plt.xlabel('时间 t')
    plt.ylabel('角动量 L')
    plt.title('角动量随时间的变化')
    plt.grid(True)
    plt.show()
    
    # 计算相对误差
    energy_error = (energies - energies[0])/energies[0]
    angular_momentum_error = (angular_momentum - angular_momentum[0])/angular_momentum[0]
    print(f'能量相对误差: {np.max(np.abs(energy_error)):.2e}')
    print(f'角动量相对误差: {np.max(np.abs(angular_momentum_error)):.2e}')

if __name__ == "__main__":
    main()
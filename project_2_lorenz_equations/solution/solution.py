#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程与确定性混沌 - 参考答案

作者：计算物理课程组
日期：2025年
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


def lorenz_system(state: np.ndarray, t: float, sigma: float, r: float, b: float) -> np.ndarray:
    """
    洛伦兹方程组的右端函数
    
    参数:
        state: [x, y, z] 当前状态向量
        t: 时间（本系统中未显式使用）
        sigma, r, b: 洛伦兹方程参数
    
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1/2, ti + h/2, *args)
        k3 = h * f(yi + k2/2, ti + h/2, *args)
        k4 = h * f(yi + k3, ti + h, *args)
        
        y[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法求解常微分方程组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1, ti + h, *args)
        
        y[i+1] = yi + (k1 + k2) / 2
    
    return t, y


def solve_lorenz_equations(sigma: float, r: float, b: float,
                          x0: float, y0: float, z0: float,
                          t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解洛伦兹方程组
    
    参数:
        sigma, r, b: 洛伦兹方程参数
        x0, y0, z0: 初始条件
        t_span: 时间范围
        dt: 时间步长
    
    返回:
        t: 时间数组
        x, y, z: 三个状态变量数组
    """
    y0_vec = np.array([x0, y0, z0])
    t, solution = runge_kutta_4(lorenz_system, y0_vec, t_span, dt, sigma, r, b)
    
    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]
    
    return t, x, y, z


def compare_methods_lorenz(sigma: float, r: float, b: float,
                          x0: float, y0: float, z0: float,
                          t_span: Tuple[float, float], dt: float) -> Dict:
    """
    比较三种数值方法求解洛伦兹方程组
    
    返回:
        包含三种方法结果的字典
    """
    y0_vec = np.array([x0, y0, z0])
    args = (sigma, r, b)
    
    # 欧拉法
    t_euler, sol_euler = euler_method(lorenz_system, y0_vec, t_span, dt, *args)
    
    # 改进欧拉法
    t_ie, sol_ie = improved_euler_method(lorenz_system, y0_vec, t_span, dt, *args)
    
    # 4阶龙格-库塔法
    t_rk4, sol_rk4 = runge_kutta_4(lorenz_system, y0_vec, t_span, dt, *args)
    
    return {
        'euler': {
            't': t_euler, 
            'x': sol_euler[:, 0], 
            'y': sol_euler[:, 1], 
            'z': sol_euler[:, 2]
        },
        'improved_euler': {
            't': t_ie, 
            'x': sol_ie[:, 0], 
            'y': sol_ie[:, 1], 
            'z': sol_ie[:, 2]
        },
        'rk4': {
            't': t_rk4, 
            'x': sol_rk4[:, 0], 
            'y': sol_rk4[:, 1], 
            'z': sol_rk4[:, 2]
        }
    }


def trajectory_distance(x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
                       x2: np.ndarray, y2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    计算两条轨道间的欧几里得距离
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def analyze_chaos_behavior(sigma: float = 10.0, r: float = 28.0, b: float = 8/3,
                          x0: float = 0.0, y0: float = 1.0, z0: float = 0.0,
                          t_span: Tuple[float, float] = (0, 30), dt: float = 0.001) -> Dict:
    """
    分析洛伦兹系统的混沌行为
    
    返回:
        包含混沌分析结果的字典
    """
    # 基准轨道
    t, x1, y1, z1 = solve_lorenz_equations(sigma, r, b, x0, y0, z0, t_span, dt)
    
    # 微小扰动的轨道
    perturbation = 1e-8
    t, x2, y2, z2 = solve_lorenz_equations(sigma, r, b, 
                                          x0 + perturbation, y0, z0, 
                                          t_span, dt)
    
    # 计算轨道间距离
    distance = trajectory_distance(x1, y1, z1, x2, y2, z2)
    
    # 估算李雅普诺夫指数（最大）
    # 在线性增长阶段拟合指数增长
    log_distance = np.log(distance + 1e-15)  # 避免log(0)
    
    # 寻找指数增长阶段（距离在合理范围内）
    valid_indices = np.where((distance > perturbation * 10) & (distance < 1.0))[0]
    
    if len(valid_indices) > 10:
        t_fit = t[valid_indices]
        log_dist_fit = log_distance[valid_indices]
        
        # 线性拟合 log(distance) vs time
        coeffs = np.polyfit(t_fit, log_dist_fit, 1)
        lyapunov_exponent = coeffs[0]
    else:
        lyapunov_exponent = np.nan
    
    # 计算轨道统计特性
    x_stats = {
        'mean': np.mean(x1),
        'std': np.std(x1),
        'min': np.min(x1),
        'max': np.max(x1)
    }
    
    y_stats = {
        'mean': np.mean(y1),
        'std': np.std(y1),
        'min': np.min(y1),
        'max': np.max(y1)
    }
    
    z_stats = {
        'mean': np.mean(z1),
        'std': np.std(z1),
        'min': np.min(z1),
        'max': np.max(z1)
    }
    
    return {
        'reference_trajectory': {'t': t, 'x': x1, 'y': y1, 'z': z1},
        'perturbed_trajectory': {'t': t, 'x': x2, 'y': y2, 'z': z2},
        'distance': distance,
        'lyapunov_exponent': lyapunov_exponent,
        'statistics': {
            'x': x_stats,
            'y': y_stats,
            'z': z_stats
        }
    }


def plot_time_series(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    title: str = "洛伦兹方程时间序列") -> None:
    """
    绘制洛伦兹方程的时间序列图
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, x, 'b-', linewidth=0.8)
    plt.ylabel('x(t)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, y, 'r-', linewidth=0.8)
    plt.ylabel('y(t)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, z, 'g-', linewidth=0.8)
    plt.xlabel('时间 t')
    plt.ylabel('z(t)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_strange_attractor(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          title: str = "洛伦兹奇异吸引子") -> None:
    """
    绘制洛伦兹奇异吸引子的3D图和投影图
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 3D轨道图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x, y, z, linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D轨道')
    
    # XY投影
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x, y, linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY投影')
    ax2.grid(True, alpha=0.3)
    
    # XZ投影
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x, z, linewidth=0.5, alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ投影')
    ax3.grid(True, alpha=0.3)
    
    # YZ投影
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(y, z, linewidth=0.5, alpha=0.8)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ投影')
    ax4.grid(True, alpha=0.3)
    
    # 庞加莱截面 (z = r-1)
    ax5 = fig.add_subplot(2, 3, 5)
    r = 28.0  # 假设使用标准参数
    z_section = r - 1
    tolerance = 0.5
    
    # 找到接近截面的点
    indices = np.where(np.abs(z - z_section) < tolerance)[0]
    if len(indices) > 0:
        ax5.scatter(x[indices], y[indices], s=1, alpha=0.6)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title(f'庞加莱截面 (z≈{z_section:.1f})')
    ax5.grid(True, alpha=0.3)
    
    # 相空间密度图
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist2d(x, y, bins=50, density=True, cmap='viridis')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('XY相空间密度')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_method_comparison_lorenz(results: Dict) -> None:
    """
    绘制不同数值方法的比较图
    """
    fig = plt.figure(figsize=(20, 15))
    
    methods = ['euler', 'improved_euler', 'rk4']
    method_names = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']
    colors = ['blue', 'orange', 'green']
    
    # 时间序列比较（只显示y分量）
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(3, 3, i+1)
        t = results[method]['t']
        y = results[method]['y']
        
        plt.plot(t, y, color=color, linewidth=1)
        plt.xlabel('时间 t')
        plt.ylabel('y(t)')
        plt.title(f'{name} - 时间序列')
        plt.grid(True, alpha=0.3)
    
    # XY相空间比较
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(3, 3, i+4)
        x = results[method]['x']
        y = results[method]['y']
        
        plt.plot(x, y, color=color, linewidth=0.5, alpha=0.8)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{name} - XY投影')
        plt.grid(True, alpha=0.3)
    
    # 3D轨道比较（子图）
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = fig.add_subplot(3, 3, i+7, projection='3d')
        x = results[method]['x']
        y = results[method]['y']
        z = results[method]['z']
        
        ax.plot(x, y, z, color=color, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name} - 3D轨道')
    
    plt.tight_layout()
    plt.show()


def plot_chaos_analysis(chaos_results: Dict) -> None:
    """
    绘制混沌行为分析图
    """
    fig = plt.figure(figsize=(15, 10))
    
    t = chaos_results['reference_trajectory']['t']
    x1 = chaos_results['reference_trajectory']['x']
    y1 = chaos_results['reference_trajectory']['y']
    x2 = chaos_results['perturbed_trajectory']['x']
    y2 = chaos_results['perturbed_trajectory']['y']
    distance = chaos_results['distance']
    
    # 轨道比较
    plt.subplot(2, 2, 1)
    plt.plot(t[:5000], y1[:5000], 'b-', label='基准轨道', linewidth=1)
    plt.plot(t[:5000], y2[:5000], 'r--', label='扰动轨道', linewidth=1, alpha=0.8)
    plt.xlabel('时间 t')
    plt.ylabel('y(t)')
    plt.title('轨道敏感性比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 距离演化
    plt.subplot(2, 2, 2)
    plt.semilogy(t, distance, 'purple', linewidth=1)
    plt.xlabel('时间 t')
    plt.ylabel('轨道间距离 (对数尺度)')
    plt.title('轨道发散分析')
    plt.grid(True, alpha=0.3)
    
    # 相空间比较
    plt.subplot(2, 2, 3)
    plt.plot(x1, y1, 'b-', linewidth=0.5, alpha=0.7, label='基准轨道')
    plt.plot(x2, y2, 'r-', linewidth=0.5, alpha=0.7, label='扰动轨道')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('相空间轨道比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 李雅普诺夫指数估算
    plt.subplot(2, 2, 4)
    log_distance = np.log(distance + 1e-15)
    plt.plot(t, log_distance, 'green', linewidth=1)
    
    # 显示拟合直线（如果李雅普诺夫指数有效）
    lyap = chaos_results['lyapunov_exponent']
    if not np.isnan(lyap):
        fit_line = lyap * t + np.log(1e-8)
        plt.plot(t, fit_line, 'k--', linewidth=2, 
                label=f'拟合斜率: {lyap:.3f}')
        plt.legend()
    
    plt.xlabel('时间 t')
    plt.ylabel('ln(距离)')
    plt.title('李雅普诺夫指数估算')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explore_parameter_space() -> None:
    """
    探索参数空间，研究不同r值下的系统行为
    """
    sigma, b = 10.0, 8/3
    x0, y0, z0 = 0.0, 1.0, 0.0
    t_span = (0, 30)
    dt = 0.001
    
    # 不同的r值
    r_values = [10, 15, 20, 24, 28, 35]
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, r in enumerate(r_values):
        t, x, y, z = solve_lorenz_equations(sigma, r, b, x0, y0, z0, t_span, dt)
        
        # 时间序列
        plt.subplot(3, len(r_values), i+1)
        plt.plot(t, y, linewidth=0.8)
        plt.title(f'r = {r}')
        plt.ylabel('y(t)' if i == 0 else '')
        if i == len(r_values)//2:
            plt.xlabel('时间序列')
        plt.grid(True, alpha=0.3)
        
        # XY相空间
        plt.subplot(3, len(r_values), i+1+len(r_values))
        plt.plot(x, y, linewidth=0.5)
        plt.ylabel('Y' if i == 0 else '')
        plt.xlabel('X')
        if i == len(r_values)//2:
            plt.xlabel('XY相空间')
        plt.grid(True, alpha=0.3)
        
        # XZ相空间
        plt.subplot(3, len(r_values), i+1+2*len(r_values))
        plt.plot(x, z, linewidth=0.5)
        plt.ylabel('Z' if i == 0 else '')
        plt.xlabel('X')
        if i == len(r_values)//2:
            plt.xlabel('XZ相空间')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('洛伦兹系统参数空间探索', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示洛伦兹方程的完整分析
    """
    # 标准参数
    sigma, r, b = 10.0, 28.0, 8/3
    x0, y0, z0 = 0.0, 1.0, 0.0
    t_span = (0, 50)
    dt = 0.001
    
    print("=== 洛伦兹方程与确定性混沌分析 ===")
    print(f"参数: σ={sigma}, r={r}, b={b:.3f}")
    print(f"初始条件: ({x0}, {y0}, {z0})")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    # 1. 基本求解
    print("\n1. 求解洛伦兹方程...")
    t, x, y, z = solve_lorenz_equations(sigma, r, b, x0, y0, z0, t_span, dt)
    
    # 2. 时间序列可视化
    print("\n2. 绘制时间序列...")
    plot_time_series(t, x, y, z)
    
    # 3. 奇异吸引子可视化
    print("\n3. 绘制奇异吸引子...")
    plot_strange_attractor(x, y, z)
    
    # 4. 方法比较
    print("\n4. 比较数值方法...")
    t_span_short = (0, 20)  # 较短时间以便比较
    results = compare_methods_lorenz(sigma, r, b, x0, y0, z0, t_span_short, dt)
    plot_method_comparison_lorenz(results)
    
    # 5. 混沌行为分析
    print("\n5. 分析混沌行为...")
    chaos_results = analyze_chaos_behavior(sigma, r, b, x0, y0, z0, (0, 30), dt)
    plot_chaos_analysis(chaos_results)
    
    # 6. 参数空间探索
    print("\n6. 探索参数空间...")
    explore_parameter_space()
    
    # 7. 数值结果统计
    print("\n7. 数值结果统计:")
    stats = chaos_results['statistics']
    print(f"X统计: 均值={stats['x']['mean']:.3f}, 标准差={stats['x']['std']:.3f}")
    print(f"Y统计: 均值={stats['y']['mean']:.3f}, 标准差={stats['y']['std']:.3f}")
    print(f"Z统计: 均值={stats['z']['mean']:.3f}, 标准差={stats['z']['std']:.3f}")
    
    lyap = chaos_results['lyapunov_exponent']
    if not np.isnan(lyap):
        print(f"估算的最大李雅普诺夫指数: {lyap:.3f}")
        print(f"理论值约为: 0.906")
    else:
        print("李雅普诺夫指数估算失败")


if __name__ == "__main__":
    main()
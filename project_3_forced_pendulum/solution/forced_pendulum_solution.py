import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, y, gamma, omega_d, F_d):
    """
    受驱单摆的常微分方程组。
    y[0]: 角度 theta
    y[1]: 角速度 omega
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -gamma * omega - np.sin(theta) + F_d * np.cos(omega_d * t)
    return [dtheta_dt, domega_dt]

def euler_method(ode_func, y0, t_span, dt, *args):
    """
    欧拉法求解ODE。
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        dy = np.array(ode_func(t[i], y[i], *args))
        y[i+1] = y[i] + dy * dt
    return t, y

def improved_euler_method(ode_func, y0, t_span, dt, *args):
    """
    改进欧拉法（Heun's method）求解ODE。
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = np.array(ode_func(t[i], y[i], *args))
        y_temp = y[i] + k1 * dt
        k2 = np.array(ode_func(t[i+1], y_temp, *args))
        y[i+1] = y[i] + 0.5 * (k1 + k2) * dt
    return t, y

def rk4_method(ode_func, y0, t_span, dt, *args):
    """
    四阶龙格-库塔法（RK4）求解ODE。
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = np.array(ode_func(t[i], y[i], *args))
        k2 = np.array(ode_func(t[i] + 0.5 * dt, y[i] + 0.5 * k1 * dt, *args))
        k3 = np.array(ode_func(t[i] + 0.5 * dt, y[i] + 0.5 * k2 * dt, *args))
        k4 = np.array(ode_func(t[i] + dt, y[i] + k3 * dt, *args))
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return t, y

def solve_and_compare_methods(ode_func, y0, t_span, dt, *args):
    """
    使用欧拉法、改进欧拉法和RK4法求解ODE，并返回结果。
    """
    t_euler, y_euler = euler_method(ode_func, y0, t_span, dt, *args)
    t_improved_euler, y_improved_euler = improved_euler_method(ode_func, y0, t_span, dt, *args)
    t_rk4, y_rk4 = rk4_method(ode_func, y0, t_span, dt, *args)
    return (t_euler, y_euler), (t_improved_euler, y_improved_euler), (t_rk4, y_rk4)

def plot_results(t_values, y_values, labels, title, y_labels=None):
    """
    绘制结果。
    """
    plt.figure(figsize=(10, 6))
    for i, (t, y) in enumerate(zip(t_values, y_values)):
        plt.plot(t, y[:, 0], label=f'{labels[i]} - Angle')
        plt.plot(t, y[:, 1], linestyle='--', label=f'{labels[i]} - Angular Velocity')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_space(y_values, labels, title):
    """
    绘制相空间图。
    """
    plt.figure(figsize=(8, 8))
    for i, y in enumerate(y_values):
        plt.plot(y[:, 0], y[:, 1], label=labels[i])
    plt.title(title)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_resonance(gamma, omega_d_range, F_d, y0, t_span, dt):
    """
    分析共振现象，绘制不同驱动频率下的稳态振幅。
    """
    amplitudes = []
    for omega_d in omega_d_range:
        sol = solve_ivp(forced_pendulum_ode, t_span, y0, args=(gamma, omega_d, F_d), dense_output=True, rtol=1e-6, atol=1e-9)
        # 取后半段数据计算稳态振幅
        t_steady = np.linspace(t_span[0] + (t_span[1] - t_span[0]) / 2, t_span[1], 100)
        y_steady = sol.sol(t_steady)
        amplitude = np.max(np.abs(y_steady[0]))
        amplitudes.append(amplitude)
    
    plt.figure(figsize=(10, 6))
    plt.plot(omega_d_range, amplitudes, 'o-')
    plt.title('Resonance Curve: Amplitude vs. Driving Frequency')
    plt.xlabel('Driving Frequency (rad/s)')
    plt.ylabel('Steady-state Amplitude (rad)')
    plt.grid(True)
    plt.show()
    return amplitudes

def analyze_chaos(gamma, omega_d, F_d_range, y0, t_span, dt):
    """
    分析混沌行为，绘制分岔图（例如，庞加莱截面）。
    这里简化为绘制不同驱动力下的相空间图，观察混沌迹象。
    """
    for F_d in F_d_range:
        t_rk4, y_rk4 = rk4_method(forced_pendulum_ode, y0, t_span, dt, gamma, omega_d, F_d)
        plot_phase_space([y_rk4], [f'F_d = {F_d}'], f'Phase Space for F_d = {F_d}')

if __name__ == '__main__':
    # 示例参数
    gamma = 0.5      # 阻尼系数
    omega_d = 0.667  # 驱动频率
    F_d = 1.2        # 驱动力幅度
    y0 = [0.0, 0.0]  # 初始条件 [theta, omega]
    t_span = [0, 100] # 模拟时间范围
    dt = 0.01        # 时间步长

    # 1. 求解并比较三种方法
    (t_e, y_e), (t_ie, y_ie), (t_rk4, y_rk4) = solve_and_compare_methods(
        forced_pendulum_ode, y0, t_span, dt, gamma, omega_d, F_d
    )

    # 绘制时间序列图
    plot_results(
        [t_e, t_ie, t_rk4],
        [y_e, y_ie, y_rk4],
        ['Euler', 'Improved Euler', 'RK4'],
        'Forced Pendulum: Angle and Angular Velocity vs. Time'
    )

    # 绘制相空间图
    plot_phase_space(
        [y_e, y_ie, y_rk4],
        ['Euler', 'Improved Euler', 'RK4'],
        'Forced Pendulum: Phase Space'
    )

    # 2. 共振分析
    omega_d_range = np.linspace(0.1, 2.0, 50)
    analyze_resonance(gamma, omega_d_range, F_d, y0, t_span, dt)

    # 3. 混沌行为分析（通过改变驱动力幅度）
    F_d_range = [0.5, 1.0, 1.2, 1.5]
    analyze_chaos(gamma, omega_d, F_d_range, y0, t_span, dt)

    print("Forced Pendulum simulation complete.")
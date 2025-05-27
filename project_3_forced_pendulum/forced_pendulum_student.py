import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, y, gamma, omega_d, F_d):
    """
    受驱单摆的常微分方程组。
    
    参数：
    t : float
        当前时间
    y : array_like
        状态向量 [theta, omega]
        theta: 角度
        omega: 角速度
    gamma : float
        阻尼系数
    omega_d : float
        驱动频率
    F_d : float
        驱动力幅度
    
    返回：
    list
        [dtheta_dt, domega_dt] - 角度和角速度的导数
    """
    theta, omega = y
    # TODO: 实现受驱单摆的微分方程组 (约 2-3 行代码)
    # 提示：
    # 1. dtheta/dt = omega
    # 2. domega/dt = -gamma * omega - sin(theta) + F_d * cos(omega_d * t)
    raise NotImplementedError("请在此实现受驱单摆的微分方程组。")

def euler_method(ode_func, y0, t_span, dt, *args):
    """
    使用欧拉法求解ODE。
    
    参数：
    ode_func : callable
        ODE函数，返回导数
    y0 : array_like
        初始条件
    t_span : tuple
        时间范围 (t_start, t_end)
    dt : float
        时间步长
    *args : tuple
        传递给ODE函数的额外参数
    
    返回：
    tuple
        (t, y) - 时间点和对应的解
    """
    # TODO: 实现欧拉法 (约 5-7 行代码)
    # 提示：
    # 1. 使用 np.arange 生成时间点数组：t = np.arange(t_span[0], t_span[1] + dt, dt)
    # 2. 初始化解向量：y = np.zeros((len(t), len(y0)))
    # 3. 设置初始条件：y[0] = y0
    # 4. 使用循环实现欧拉法迭代：y[i+1] = y[i] + dy * dt
    raise NotImplementedError("请在此实现欧拉法。")

def improved_euler_method(ode_func, y0, t_span, dt, *args):
    """
    使用改进的欧拉法（Heun方法）求解ODE。
    
    参数：
    ode_func : callable
        ODE函数，返回导数
    y0 : array_like
        初始条件
    t_span : tuple
        时间范围 (t_start, t_end)
    dt : float
        时间步长
    *args : tuple
        传递给ODE函数的额外参数
    
    返回：
    tuple
        (t, y) - 时间点和对应的解
    """
    # TODO: 实现改进的欧拉法 (约 8-10 行代码)
    # 提示：
    # 1. 生成时间点数组
    # 2. 初始化解向量并设置初始条件
    # 3. 在循环中：
    #    - 计算预测值：k1 = ode_func(t[i], y[i], *args)
    #    - 计算中点预测：y_temp = y[i] + k1 * dt
    #    - 计算校正值：k2 = ode_func(t[i+1], y_temp, *args)
    #    - 更新解：y[i+1] = y[i] + 0.5 * (k1 + k2) * dt
    raise NotImplementedError("请在此实现改进的欧拉法。")

def rk4_method(ode_func, y0, t_span, dt, *args):
    """
    使用四阶龙格-库塔法（RK4）求解ODE。
    
    参数：
    ode_func : callable
        ODE函数，返回导数
    y0 : array_like
        初始条件
    t_span : tuple
        时间范围 (t_start, t_end)
    dt : float
        时间步长
    *args : tuple
        传递给ODE函数的额外参数
    
    返回：
    tuple
        (t, y) - 时间点和对应的解
    """
    # TODO: 实现四阶龙格-库塔法 (约 10-12 行代码)
    # 提示：
    # 1. 生成时间点数组
    # 2. 初始化解向量并设置初始条件
    # 3. 在循环中计算四个斜率：
    #    k1 = f(t[i], y[i])
    #    k2 = f(t[i] + dt/2, y[i] + k1*dt/2)
    #    k3 = f(t[i] + dt/2, y[i] + k2*dt/2)
    #    k4 = f(t[i] + dt, y[i] + k3*dt)
    # 4. 更新解：y[i+1] = y[i] + (k1 + 2k2 + 2k3 + k4)*dt/6
    raise NotImplementedError("请在此实现四阶龙格-库塔法。")

def plot_results(t_values, y_values, labels, title):
    """
    绘制结果。
    
    参数：
    t_values : list of array_like
        时间数组列表
    y_values : list of array_like
        解向量列表
    labels : list of str
        图例标签列表
    title : str
        图标题
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
    
    参数：
    y_values : list of array_like
        解向量列表
    labels : list of str
        图例标签列表
    title : str
        图标题
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
    
    参数：
    gamma : float
        阻尼系数
    omega_d_range : array_like
        驱动频率范围
    F_d : float
        驱动力幅度
    y0 : array_like
        初始条件
    t_span : tuple
        时间范围
    dt : float
        时间步长
    
    返回：
    list
        不同驱动频率下的振幅
    """
    # TODO: 实现共振分析 (约 15-20 行代码)
    # 提示：
    # 1. 对每个驱动频率：
    #    - 使用RK4方法求解方程
    #    - 取后半段数据计算稳态振幅
    #    - 记录最大振幅
    # 2. 绘制驱动频率-振幅关系图
    raise NotImplementedError("请在此实现共振分析。")

def analyze_chaos(gamma, omega_d, F_d_range, y0, t_span, dt):
    """
    分析混沌行为，绘制不同驱动力下的相空间图。
    
    参数：
    gamma : float
        阻尼系数
    omega_d : float
        驱动频率
    F_d_range : array_like
        驱动力幅度范围
    y0 : array_like
        初始条件
    t_span : tuple
        时间范围
    dt : float
        时间步长
    """
    # TODO: 实现混沌分析 (约 10-15 行代码)
    # 提示：
    # 1. 对每个驱动力幅度：
    #    - 使用RK4方法求解方程
    #    - 绘制相空间图
    #    - 观察轨道的规律性/混沌性
    raise NotImplementedError("请在此实现混沌分析。")

if __name__ == '__main__':
    # 示例参数
    gamma = 0.5      # 阻尼系数
    omega_d = 0.667  # 驱动频率
    F_d = 1.2        # 驱动力幅度
    y0 = [0.0, 0.0]  # 初始条件 [theta, omega]
    t_span = [0, 100] # 模拟时间范围
    dt = 0.01        # 时间步长

    # TODO: 在此处调用上述函数完成实验任务
    # 1. 使用三种数值方法求解方程并比较结果
    # 2. 分析共振现象
    # 3. 研究混沌行为
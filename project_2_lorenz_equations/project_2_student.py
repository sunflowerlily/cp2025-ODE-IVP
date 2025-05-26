#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程与确定性混沌 - 学生代码模板

学生姓名：[请填写您的姓名]
学号：[请填写您的学号]
完成日期：[请填写完成日期]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lorenz_system(state: np.ndarray, t: float, sigma: float, r: float, b: float) -> np.ndarray:
    """
    洛伦兹方程组的右端函数
    
    方程组：
    dx/dt = σ(y - x)
    dy/dt = rx - y - xz
    dz/dt = xy - bz
    
    参数:
        state: np.ndarray, 形状为(3,), 当前状态向量 [x, y, z]
        t: float, 时间（本系统中未显式使用）
        sigma: float, 普朗特数
        r: float, 瑞利数
        b: float, 几何因子
    
    返回:
        np.ndarray, 形状为(3,), 导数向量 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    
    # TODO: 实现洛伦兹方程组 (约3-4行代码)
    # 提示：根据上面的方程组计算 dx/dt, dy/dt, dz/dt
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 lorenz_system 函数中实现洛伦兹方程组")
    
    return np.array([dxdt, dydt, dzdt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现欧拉法迭代 (约3-5行代码)
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        raise NotImplementedError("请在 euler_method 函数中实现欧拉法迭代")
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现改进欧拉法 (约6-8行代码)
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        raise NotImplementedError("请在 improved_euler_method 函数中实现改进欧拉法")
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现4阶龙格-库塔法 (约8-12行代码)
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        raise NotImplementedError("请在 runge_kutta_4 函数中实现4阶龙格-库塔法")
    
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
        t: np.ndarray, 时间数组
        x, y, z: np.ndarray, 三个状态变量数组
    """
    # TODO: 调用runge_kutta_4函数求解洛伦兹方程组 (约3-5行代码)
    # 提示：
    # 1. 构造初始条件向量 y0_vec = [x0, y0, z0]
    # 2. 调用runge_kutta_4函数
    # 3. 从解中提取x, y, z分量
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 solve_lorenz_equations 函数中实现求解逻辑")
    
    return t, x, y, z


def compare_methods_lorenz(sigma: float, r: float, b: float,
                          x0: float, y0: float, z0: float,
                          t_span: Tuple[float, float], dt: float) -> Dict:
    """
    比较三种数值方法求解洛伦兹方程组
    
    参数:
        sigma, r, b: 洛伦兹方程参数
        x0, y0, z0: 初始条件
        t_span: 时间范围
        dt: 时间步长
    
    返回:
        Dict: 包含三种方法结果的字典
    """
    # TODO: 使用三种方法求解并返回结果字典 (约15-20行代码)
    # 提示：
    # 1. 构造初始条件向量和参数
    # 2. 分别调用三种方法
    # 3. 构造并返回结果字典
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 compare_methods_lorenz 函数中实现方法比较")
    
    return results


def trajectory_distance(x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
                       x2: np.ndarray, y2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    计算两条轨道间的欧几里得距离
    
    参数:
        x1, y1, z1: 第一条轨道的坐标
        x2, y2, z2: 第二条轨道的坐标
    
    返回:
        np.ndarray: 轨道间距离数组
    """
    # TODO: 计算欧几里得距离 (约1-2行代码)
    # 提示：distance = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 trajectory_distance 函数中实现距离计算")
    
    return distance


def analyze_chaos_behavior(sigma: float = 10.0, r: float = 28.0, b: float = 8/3,
                          x0: float = 0.0, y0: float = 1.0, z0: float = 0.0,
                          t_span: Tuple[float, float] = (0, 30), dt: float = 0.001) -> Dict:
    """
    分析洛伦兹系统的混沌行为
    
    分析内容：
    1. 对初始条件的敏感依赖性
    2. 李雅普诺夫指数估算
    3. 轨道统计特性
    
    返回:
        Dict: 包含混沌分析结果的字典
    """
    # TODO: 实现混沌行为分析 (约30-40行代码)
    # 提示：
    # 1. 求解基准轨道
    # 2. 求解微小扰动的轨道
    # 3. 计算轨道间距离
    # 4. 估算李雅普诺夫指数
    # 5. 计算统计特性
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 analyze_chaos_behavior 函数中实现混沌分析")
    
    return results


def plot_time_series(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    title: str = "洛伦兹方程时间序列") -> None:
    """
    绘制洛伦兹方程的时间序列图
    
    参数:
        t: 时间数组
        x, y, z: 状态变量数组
        title: 图标题
    """
    # TODO: 绘制三个子图显示x(t), y(t), z(t) (约15-20行代码)
    # 提示：使用plt.subplot(3, 1, i)创建子图
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 plot_time_series 函数中实现时间序列绘图")


def plot_strange_attractor(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          title: str = "洛伦兹奇异吸引子") -> None:
    """
    绘制洛伦兹奇异吸引子的3D图和投影图
    
    参数:
        x, y, z: 状态变量数组
        title: 图标题
    """
    # TODO: 绘制奇异吸引子图 (约25-35行代码)
    # 提示：
    # 1. 创建2x3子图布局
    # 2. 绘制3D轨道图
    # 3. 绘制XY, XZ, YZ投影
    # 4. 绘制庞加莱截面
    # 5. 绘制相空间密度图
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 plot_strange_attractor 函数中实现奇异吸引子绘图")


def plot_method_comparison_lorenz(results: Dict) -> None:
    """
    绘制不同数值方法的比较图
    
    参数:
        results: compare_methods_lorenz函数的返回结果
    """
    # TODO: 绘制方法比较图 (约25-35行代码)
    # 提示：
    # 1. 创建3x3子图布局
    # 2. 上排：三种方法的时间序列
    # 3. 中排：三种方法的XY投影
    # 4. 下排：三种方法的3D轨道
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 plot_method_comparison_lorenz 函数中实现比较图绘制")


def plot_chaos_analysis(chaos_results: Dict) -> None:
    """
    绘制混沌行为分析图
    
    参数:
        chaos_results: analyze_chaos_behavior函数的返回结果
    """
    # TODO: 绘制混沌分析图 (约25-35行代码)
    # 提示：
    # 1. 创建2x2子图布局
    # 2. 轨道敏感性比较
    # 3. 距离演化（对数尺度）
    # 4. 相空间轨道比较
    # 5. 李雅普诺夫指数估算
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 plot_chaos_analysis 函数中实现混沌分析图绘制")


def explore_parameter_space() -> None:
    """
    探索参数空间，研究不同r值下的系统行为
    
    分析内容：
    1. 不同r值的时间序列
    2. 相空间轨道变化
    3. 从周期到混沌的转变
    """
    # TODO: 实现参数空间探索 (约30-40行代码)
    # 提示：
    # 1. 设置不同的r值
    # 2. 对每个r值求解洛伦兹方程
    # 3. 绘制时间序列和相空间图
    # 4. 观察系统行为的变化
    # [STUDENT_CODE_HERE]
    raise NotImplementedError("请在 explore_parameter_space 函数中实现参数空间探索")


def main():
    """
    主函数：演示洛伦兹方程的完整分析
    
    执行步骤：
    1. 基本求解和可视化
    2. 数值方法比较
    3. 混沌行为分析
    4. 参数空间探索
    5. 结果统计
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
    
    try:
        # TODO: 实现主函数逻辑 (约20-30行代码)
        # 1. 基本求解
        print("\n1. 求解洛伦兹方程...")
        # [STUDENT_CODE_HERE]
        
        # 2. 时间序列可视化
        print("\n2. 绘制时间序列...")
        # [STUDENT_CODE_HERE]
        
        # 3. 奇异吸引子可视化
        print("\n3. 绘制奇异吸引子...")
        # [STUDENT_CODE_HERE]
        
        # 4. 方法比较
        print("\n4. 比较数值方法...")
        # [STUDENT_CODE_HERE]
        
        # 5. 混沌行为分析
        print("\n5. 分析混沌行为...")
        # [STUDENT_CODE_HERE]
        
        # 6. 参数空间探索
        print("\n6. 探索参数空间...")
        # [STUDENT_CODE_HERE]
        
        # 7. 数值结果统计
        print("\n7. 数值结果统计:")
        # [STUDENT_CODE_HERE]
        
    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行主程序。")


if __name__ == "__main__":
    main()
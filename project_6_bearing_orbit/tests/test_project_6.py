import numpy as np
import unittest
from scipy.integrate import solve_ivp
from project_6_student import (
    bearing_orbit_ode,
    rk4_step,
    solve_ode,
    calculate_energy,
    calculate_angular_momentum,
    analyze_precession
)

class TestBearingOrbit(unittest.TestCase):
    def setUp(self):
        """设置测试参数"""
        self.G = 1.0
        self.M = 10.0
        self.L = 2.0
        self.dt = 0.001
        self.t_span = (0, 20)
        self.initial_state = np.array([1.0, 0.0, 0.0, 1.0])

    def test_bearing_orbit_ode(self):
        """测试轴承轨道ODE函数"""
        state = np.array([1.0, 1.0, 0.5, -0.5])
        t = 0.0
        derivative = bearing_orbit_ode(state, t, self.G, self.M, self.L)
        
        # 验证导数的形状
        self.assertEqual(derivative.shape, (4,))
        
        # 验证速度分量
        np.testing.assert_almost_equal(derivative[0], state[2])
        np.testing.assert_almost_equal(derivative[1], state[3])
        
        # 验证加速度分量的符号和大小级别
        r = np.sqrt(state[0]**2 + state[1]**2)
        factor = self.G * self.M / (r**2 * np.sqrt(r**2 + self.L**2/4))
        self.assertLess(abs(derivative[2] + factor * state[0]), 1e-10)
        self.assertLess(abs(derivative[3] + factor * state[1]), 1e-10)

    def test_rk4_step_accuracy(self):
        """测试RK4方法的精度"""
        # 使用简单的线性ODE进行测试
        def test_ode(y, t):
            return np.array([y[0]])

        initial_state = np.array([1.0])
        t = 0.0
        dt = 0.1

        # 计算RK4结果
        rk4_result = rk4_step(test_ode, initial_state, t, dt)

        # 计算解析解
        exact_solution = initial_state * np.exp(dt)

        # 验证RK4方法的精度（应该是4阶精度）
        error = np.abs(rk4_result - exact_solution)
        self.assertLess(error[0], 1e-5)

    def test_solve_ode_basic_properties(self):
        """测试ODE求解器的基本属性"""
        t, states = solve_ode(
            bearing_orbit_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            G=self.G,
            M=self.M,
            L=self.L
        )

        # 验证解的基本特性
        self.assertEqual(len(t), len(states))
        self.assertEqual(states.shape[1], 4)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_energy_conservation(self):
        """测试能量守恒"""
        t, states = solve_ode(
            bearing_orbit_ode,
            self.initial_state,
            (0, 1),  # 使用较短的时间以加快测试
            self.dt,
            G=self.G,
            M=self.M,
            L=self.L
        )

        # 计算初始能量
        initial_energy = calculate_energy(self.initial_state, self.G, self.M, self.L)

        # 验证能量守恒
        for state in states:
            energy = calculate_energy(state, self.G, self.M, self.L)
            relative_error = abs((energy - initial_energy) / initial_energy)
            self.assertLess(relative_error, 1e-3)

    def test_angular_momentum_conservation(self):
        """测试角动量守恒"""
        t, states = solve_ode(
            bearing_orbit_ode,
            self.initial_state,
            (0, 1),  # 使用较短的时间以加快测试
            self.dt,
            G=self.G,
            M=self.M,
            L=self.L
        )

        # 计算初始角动量
        initial_L = calculate_angular_momentum(self.initial_state)

        # 验证角动量守恒
        for state in states:
            L = calculate_angular_momentum(state)
            relative_error = abs((L - initial_L) / initial_L)
            self.assertLess(relative_error, 1e-3)

    def test_precession_analysis(self):
        """测试进动分析"""
        # 生成测试数据：模拟简单的圆周运动加进动
        t = np.linspace(0, 10, 1000)
        omega = 1.0  # 基本角频率
        precession_rate = 0.1  # 进动角速度
        
        # 生成带进动的轨道数据
        r = 1.0
        x = r * np.cos(omega * t) * np.cos(precession_rate * t) - \
            r * np.sin(omega * t) * np.sin(precession_rate * t)
        y = r * np.cos(omega * t) * np.sin(precession_rate * t) + \
            r * np.sin(omega * t) * np.cos(precession_rate * t)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        
        states = np.column_stack((x, y, vx, vy))

        # 分析进动
        calculated_rate = analyze_precession(states)

        # 验证计算的进动率是否接近设定值
        self.assertAlmostEqual(calculated_rate, precession_rate, places=1)

    def test_solution_comparison_with_scipy(self):
        """将数值解与scipy的解进行比较"""
        def scipy_bearing_orbit(t, y):
            return bearing_orbit_ode(y, t, self.G, self.M, self.L)

        # 使用scipy求解
        scipy_sol = solve_ivp(
            scipy_bearing_orbit,
            self.t_span,
            self.initial_state,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )

        # 使用学生实现求解
        t, states = solve_ode(
            bearing_orbit_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            G=self.G,
            M=self.M,
            L=self.L
        )

        # 在几个时间点比较结果
        test_times = [1.0, 5.0, 10.0]
        for test_time in test_times:
            # 获取scipy解在测试时间点的值
            scipy_idx = np.argmin(np.abs(scipy_sol.t - test_time))
            scipy_state = scipy_sol.y[:, scipy_idx]

            # 获取学生解在测试时间点的值
            student_idx = np.argmin(np.abs(t - test_time))
            student_state = states[student_idx]

            # 比较结果（允许有小的误差）
            np.testing.assert_allclose(student_state, scipy_state, rtol=1e-2)

if __name__ == '__main__':
    unittest.main()
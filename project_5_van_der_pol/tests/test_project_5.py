import numpy as np
import unittest
from scipy.integrate import solve_ivp
from project_5_student import (
    van_der_pol_ode,
    rk4_step,
    solve_ode,
    calculate_energy,
    analyze_limit_cycle
)

class TestVanDerPolOscillator(unittest.TestCase):
    def setUp(self):
        """设置测试参数"""
        self.mu = 1.0
        self.omega = 1.0
        self.dt = 0.01
        self.t_span = (0, 20)
        self.initial_state = np.array([1.0, 0.0])

    def test_van_der_pol_ode(self):
        """测试van der Pol ODE函数"""
        state = np.array([1.0, 2.0])
        t = 0.0
        derivative = van_der_pol_ode(state, t, self.mu, self.omega)
        
        # 验证导数的形状
        self.assertEqual(derivative.shape, (2,))
        
        # 验证导数的值
        expected_dx = 2.0  # v
        expected_dv = self.mu * (1 - 1.0) * 2.0 - self.omega**2 * 1.0  # mu(1-x^2)v - omega^2*x
        np.testing.assert_almost_equal(derivative[0], expected_dx)
        np.testing.assert_almost_equal(derivative[1], expected_dv)

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
            van_der_pol_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            mu=self.mu,
            omega=self.omega
        )

        # 验证解的基本特性
        self.assertEqual(len(t), len(states))
        self.assertEqual(states.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_energy_calculation(self):
        """测试能量计算函数"""
        state = np.array([1.0, 2.0])
        energy = calculate_energy(state, self.omega)

        # 验证能量计算
        expected_energy = 0.5 * 2.0**2 + 0.5 * self.omega**2 * 1.0**2
        np.testing.assert_almost_equal(energy, expected_energy)

    def test_limit_cycle_analysis(self):
        """测试极限环分析"""
        # 生成测试数据：模拟极限环行为
        t = np.linspace(0, 50, 5000)
        x = 2 * np.cos(t)
        v = -2 * np.sin(t)
        states = np.column_stack((x, v))

        # 分析极限环
        amplitude, period = analyze_limit_cycle(states)

        # 验证结果
        self.assertGreater(amplitude, 0)
        self.assertGreater(period, 0)
        np.testing.assert_almost_equal(amplitude, 2.0, decimal=1)
        np.testing.assert_almost_equal(period, 2*np.pi, decimal=1)

    def test_solution_comparison_with_scipy(self):
        """将数值解与scipy的解进行比较"""
        def scipy_van_der_pol(t, y):
            return van_der_pol_ode(y, t, self.mu, self.omega)

        # 使用scipy求解
        scipy_sol = solve_ivp(
            scipy_van_der_pol,
            self.t_span,
            self.initial_state,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )

        # 使用学生实现求解
        t, states = solve_ode(
            van_der_pol_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            mu=self.mu,
            omega=self.omega
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
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from project_3_student import (
    forced_pendulum_ode,
    euler_method,
    improved_euler_method,
    rk4_method
)

class TestProject3(unittest.TestCase):
    def setUp(self):
        """设置测试参数"""
        self.gamma = 0.5
        self.omega_d = 0.667
        self.F_d = 1.2
        self.y0 = [0.0, 0.0]
        self.t_span = [0, 10]
        self.dt = 0.01
        self.rtol = 1e-3  # 相对误差容限
        self.atol = 1e-6  # 绝对误差容限

    def test_forced_pendulum_ode(self):
        """测试受驱单摆ODE函数"""
        t = 0.0
        y = [0.1, 0.2]
        dy = forced_pendulum_ode(t, y, self.gamma, self.omega_d, self.F_d)
        
        # 验证返回值类型和长度
        self.assertIsInstance(dy, list)
        self.assertEqual(len(dy), 2)
        
        # 验证方程正确性
        expected_dtheta_dt = y[1]
        expected_domega_dt = -self.gamma * y[1] - np.sin(y[0]) + self.F_d * np.cos(self.omega_d * t)
        
        self.assertAlmostEqual(dy[0], expected_dtheta_dt, places=6)
        self.assertAlmostEqual(dy[1], expected_domega_dt, places=6)

    def get_reference_solution(self):
        """使用scipy.integrate.solve_ivp获取参考解"""
        t_eval = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        sol = solve_ivp(
            forced_pendulum_ode,
            self.t_span,
            self.y0,
            args=(self.gamma, self.omega_d, self.F_d),
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10
        )
        return sol.t, sol.y.T

    def test_euler_method(self):
        """测试欧拉法"""
        # 获取参考解
        t_ref, y_ref = self.get_reference_solution()
        
        # 获取欧拉法解
        t_euler, y_euler = euler_method(
            forced_pendulum_ode,
            self.y0,
            self.t_span,
            self.dt,
            self.gamma,
            self.omega_d,
            self.F_d
        )
        
        # 验证时间点
        np.testing.assert_allclose(t_euler, t_ref, rtol=self.rtol, atol=self.atol)
        
        # 验证解的形状
        self.assertEqual(y_euler.shape, y_ref.shape)
        
        # 验证欧拉法的收敛阶（应为1阶）
        dt_test = [0.1, 0.05, 0.025]
        errors = []
        for dt in dt_test:
            _, y = euler_method(
                forced_pendulum_ode,
                self.y0,
                [0, 1],  # 使用较短的时间范围以加快测试
                dt,
                self.gamma,
                self.omega_d,
                self.F_d
            )
            errors.append(np.max(np.abs(y[-1] - y_ref[int(1/dt)])))
        
        # 计算收敛阶
        order = np.log2(errors[0]/errors[1])
        self.assertGreater(order, 0.9)  # 应接近1

    def test_improved_euler_method(self):
        """测试改进欧拉法"""
        # 获取参考解
        t_ref, y_ref = self.get_reference_solution()
        
        # 获取改进欧拉法解
        t_improved, y_improved = improved_euler_method(
            forced_pendulum_ode,
            self.y0,
            self.t_span,
            self.dt,
            self.gamma,
            self.omega_d,
            self.F_d
        )
        
        # 验证时间点
        np.testing.assert_allclose(t_improved, t_ref, rtol=self.rtol, atol=self.atol)
        
        # 验证解的形状
        self.assertEqual(y_improved.shape, y_ref.shape)
        
        # 验证改进欧拉法的收敛阶（应为2阶）
        dt_test = [0.1, 0.05, 0.025]
        errors = []
        for dt in dt_test:
            _, y = improved_euler_method(
                forced_pendulum_ode,
                self.y0,
                [0, 1],
                dt,
                self.gamma,
                self.omega_d,
                self.F_d
            )
            errors.append(np.max(np.abs(y[-1] - y_ref[int(1/dt)])))
        
        # 计算收敛阶
        order = np.log2(errors[0]/errors[1])
        self.assertGreater(order, 1.9)  # 应接近2

    def test_rk4_method(self):
        """测试RK4方法"""
        # 获取参考解
        t_ref, y_ref = self.get_reference_solution()
        
        # 获取RK4解
        t_rk4, y_rk4 = rk4_method(
            forced_pendulum_ode,
            self.y0,
            self.t_span,
            self.dt,
            self.gamma,
            self.omega_d,
            self.F_d
        )
        
        # 验证时间点
        np.testing.assert_allclose(t_rk4, t_ref, rtol=self.rtol, atol=self.atol)
        
        # 验证解的形状
        self.assertEqual(y_rk4.shape, y_ref.shape)
        
        # 验证RK4的收敛阶（应为4阶）
        dt_test = [0.1, 0.05, 0.025]
        errors = []
        for dt in dt_test:
            _, y = rk4_method(
                forced_pendulum_ode,
                self.y0,
                [0, 1],
                dt,
                self.gamma,
                self.omega_d,
                self.F_d
            )
            errors.append(np.max(np.abs(y[-1] - y_ref[int(1/dt)])))
        
        # 计算收敛阶
        order = np.log2(errors[0]/errors[1])
        self.assertGreater(order, 3.9)  # 应接近4

if __name__ == '__main__':
    unittest.main()
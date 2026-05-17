"""物理模型測試"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics_models import (
    projectile_trajectory, spring_mass_system, simulate_spring_mass,
    heat_conduction_1d, heat_conduction_2d,
    inverted_pendulum_linearized, controllability_matrix, observability_matrix,
    fem_bar_1d, markov_steady_state, truss_2d,
)


class TestProjectile:
    """拋體運動"""

    def test_horizontal_launch(self):
        """水平拋射：水平速度恆定"""
        t, x, y, vx, vy = projectile_trajectory(10, 0, g=9.81)
        assert_allclose(vx, 10.0 * np.ones_like(vx), atol=1e-10)

    def test_45_degree_max_range(self):
        """45° 仰角射程最大"""
        v0 = 20
        _, x_45, _, _, _ = projectile_trajectory(v0, np.pi/4)
        _, x_30, _, _, _ = projectile_trajectory(v0, np.pi/6)
        _, x_60, _, _, _ = projectile_trajectory(v0, np.pi/3)
        assert x_45[-1] >= x_30[-1]
        assert x_45[-1] >= x_60[-1]

    def test_symmetry(self):
        """拋體軌跡左右對稱"""
        t, x, y, vx, vy = projectile_trajectory(20, np.pi/4)
        max_idx = np.argmax(y)
        # 上升段和下降段的 y 應大致對稱
        n_before = max_idx
        n_after = len(y) - max_idx - 1
        n = min(n_before, n_after)
        if n > 2:
            assert_allclose(y[max_idx - n:max_idx],
                            y[max_idx + 1:max_idx + n + 1][::-1], atol=0.5)


class TestSpringMass:
    """彈簧-質量系統"""

    def test_single_dof_frequency(self):
        """單自由度：f = (1/2π)√(k/m)"""
        m, k = 1.0, 100.0
        M, K, freqs, modes = spring_mass_system([m], [(-1, 0, k)])
        f_analytical = np.sqrt(k / m) / (2 * np.pi)
        assert freqs[0] == pytest.approx(f_analytical, rel=1e-6)

    def test_two_dof_frequencies(self):
        """2-DOF 系統：兩個固有頻率"""
        m1, m2 = 1.0, 1.0
        k1, k2, k3 = 100.0, 200.0, 100.0
        springs = [(-1, 0, k1), (0, 1, k2), (1, -1, k3)]
        M, K, freqs, modes = spring_mass_system([m1, m2], springs)
        assert len(freqs) == 2
        assert freqs[0] < freqs[1]  # 頻率遞增

    def test_eigenvalue_verification(self):
        """K v = λ M v"""
        masses = [2.0, 3.0, 1.0]
        springs = [(-1, 0, 50), (0, 1, 80), (1, 2, 60), (2, -1, 40)]
        M, K, freqs, modes = spring_mass_system(masses, springs)

        for i in range(len(masses)):
            lam = (2 * np.pi * freqs[i]) ** 2
            lhs = K @ modes[:, i]
            rhs = lam * M @ modes[:, i]
            assert_allclose(lhs, rhs, atol=1e-8)

    def test_simulation_energy_conservation(self):
        """無阻尼系統能量守恆"""
        M_mat = np.diag([1.0, 1.0])
        K_mat = np.array([[200, -100], [-100, 200]])
        x0 = np.array([0.01, 0.0])
        v0 = np.array([0.0, 0.0])
        t, x = simulate_spring_mass(M_mat, K_mat, x0, v0, (0, 1.0))

        # 計算各時刻總能量
        v = np.gradient(x, t, axis=1)
        KE = 0.5 * np.sum(v * (M_mat @ v), axis=0)
        PE = 0.5 * np.sum(x * (K_mat @ x), axis=0)
        E_total = KE + PE
        # 能量應幾乎恆定
        assert np.std(E_total) / np.mean(E_total) < 0.01


class TestHeatConduction:
    """熱傳導"""

    def test_1d_linear_profile(self):
        """無內熱源時，溫度分佈應為線性"""
        x, T, K = heat_conduction_1d(10, T_left=100, T_right=0)
        # 線性：T(x) = 100 - 100*x/L
        T_expected = 100 * (1 - x)
        assert_allclose(T, T_expected, atol=1e-8)

    def test_1d_symmetric_matrix(self):
        """導熱矩陣應為對稱矩陣"""
        _, _, K = heat_conduction_1d(5, 200, 20)
        assert_allclose(K, K.T, atol=1e-15)

    def test_1d_positive_definite(self):
        """導熱矩陣應為正定矩陣"""
        _, _, K = heat_conduction_1d(5, 200, 20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert all(eigenvalues > 0)

    def test_2d_boundary_satisfied(self):
        """2D 邊界條件滿足"""
        T_bc = {'top': 100, 'bottom': 0, 'left': 50, 'right': 50}
        X, Y, T = heat_conduction_2d(5, 5, T_bc)
        assert_allclose(T[-1, 1:-1], 100, atol=1)  # top
        assert_allclose(T[0, 1:-1], 0, atol=1)     # bottom


class TestInvertedPendulum:
    """倒擺控制"""

    def test_unstable_eigenvalue(self):
        """倒擺線性化後存在正實部特徵值（不穩定）"""
        A, B, C, D = inverted_pendulum_linearized()
        eigenvalues = np.linalg.eigvals(A)
        assert any(np.real(eigenvalues) > 0), "倒擺應有不穩定極點"

    def test_controllable(self):
        """倒擺系統可控"""
        A, B, C, D = inverted_pendulum_linearized()
        Ctrb = controllability_matrix(A, B)
        assert np.linalg.matrix_rank(Ctrb) == A.shape[0]

    def test_observable(self):
        """倒擺系統可觀"""
        A, B, C, D = inverted_pendulum_linearized()
        Obsv = observability_matrix(A, C)
        assert np.linalg.matrix_rank(Obsv) == A.shape[0]


class TestFEM:
    """有限元素法"""

    def test_single_bar(self):
        """單根桿件：解析解"""
        # 固定左端，右端施力 F
        nodes = [0.0, 1.0]
        elements = [(0, 1)]
        EA = [100.0]
        boundary = {0: 0.0}
        loads = {1: 10.0}
        u, stress, K = fem_bar_1d(nodes, elements, EA, boundary, loads)
        # u[1] = FL/EA = 10*1/100 = 0.1
        assert u[1] == pytest.approx(0.1)

    def test_two_bars_series(self):
        """兩根串聯桿件"""
        nodes = [0.0, 0.5, 1.0]
        elements = [(0, 1), (1, 2)]
        EA = [100.0, 100.0]
        boundary = {0: 0.0}
        loads = {2: 10.0}
        u, stress, K = fem_bar_1d(nodes, elements, EA, boundary, loads)
        # 總伸長 = F*L_total/EA = 10*1/100 = 0.1
        assert u[2] == pytest.approx(0.1)

    def test_stiffness_matrix_symmetric(self):
        """全域剛度矩陣對稱"""
        nodes = [0.0, 1.0, 2.0, 3.0]
        elements = [(0, 1), (1, 2), (2, 3)]
        EA = [100, 200, 150]
        boundary = {0: 0.0}
        loads = {3: 5.0}
        _, _, K = fem_bar_1d(nodes, elements, EA, boundary, loads)
        assert_allclose(K, K.T, atol=1e-15)


class TestMarkov:
    """Markov 鏈"""

    def test_steady_state_is_eigenvector(self):
        """穩態分佈是特徵值 1 對應的特徵向量"""
        P = np.array([[0.7, 0.2, 0.1],
                       [0.3, 0.5, 0.2],
                       [0.2, 0.3, 0.5]])
        pi = markov_steady_state(P)
        # P^T π = π
        assert_allclose(P.T @ pi, pi, atol=1e-10)

    def test_steady_state_sums_to_one(self):
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        pi = markov_steady_state(P)
        assert pi.sum() == pytest.approx(1.0)
        assert all(p >= 0 for p in pi)

    def test_convergence(self):
        """長時間演化收斂到穩態"""
        P = np.array([[0.7, 0.2, 0.1],
                       [0.3, 0.5, 0.2],
                       [0.2, 0.3, 0.5]])
        pi = markov_steady_state(P)
        # 任意初始分佈
        state = np.array([1.0, 0.0, 0.0])
        for _ in range(100):
            state = P.T @ state
        assert_allclose(state, pi, atol=1e-6)


class TestTruss:
    """桁架分析"""

    def test_simple_truss(self):
        """簡單三角桁架"""
        nodes = {0: (0, 0), 1: (1, 0), 2: (0.5, 1)}
        elements = [(0, 1), (1, 2), (0, 2)]
        supports = {0: (True, True), 1: (False, True)}
        loads = {2: (0, -1)}
        displacements, forces = truss_2d(nodes, elements, supports, loads)
        # 垂直力平衡：支承反力之和 = 外力
        # 位移應該合理（有限值）
        assert all(np.isfinite(d) for d in displacements[2])

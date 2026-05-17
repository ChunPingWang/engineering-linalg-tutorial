"""Module 3 & 4 — 線性方程組、向量空間測試"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg as sla

from src.linalg_utils import (
    gaussian_elimination, gram_schmidt, projection_onto_subspace,
)


class TestGaussianElimination:
    """3.2 高斯消去法"""

    def test_simple_system(self):
        """2x2 系統"""
        A = np.array([[2, 1], [5, 7]])
        b = np.array([11, 13])
        x = gaussian_elimination(A, b)
        assert_allclose(x, np.linalg.solve(A, b), atol=1e-10)

    def test_3x3_system(self):
        A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = np.array([8, -11, -3])
        x = gaussian_elimination(A, b)
        assert_allclose(A @ x, b, atol=1e-10)

    def test_pivoting_needed(self):
        """需要 Partial Pivoting 的情況"""
        A = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=float)
        b = np.array([2, 2, 2], dtype=float)
        x = gaussian_elimination(A, b)
        assert_allclose(A @ x, b, atol=1e-10)

    def test_singular_matrix_raises(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        b = np.array([3, 6], dtype=float)
        with pytest.raises(ValueError):
            gaussian_elimination(A, b)

    def test_large_system(self):
        """較大系統（10x10）"""
        rng = np.random.default_rng(42)
        n = 10
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        x = gaussian_elimination(A, b)
        assert_allclose(A @ x, b, atol=1e-8)

    def test_truss_statics(self):
        """靜力學驗證：合力為零"""
        # 簡化桁架：2個未知力 + 已知外力
        # F1*cos30 + F2*cos60 = 100 (水平平衡)
        # F1*sin30 + F2*sin60 = 0   (垂直平衡，假設)
        A = np.array([
            [np.cos(np.pi/6), np.cos(np.pi/3)],
            [np.sin(np.pi/6), np.sin(np.pi/3)],
        ])
        b = np.array([100, 50])
        x = gaussian_elimination(A, b)
        # 驗證殘差
        assert np.linalg.norm(A @ x - b) < 1e-10


class TestGramSchmidt:
    """4.4 Gram-Schmidt 正交化"""

    def test_2d_orthogonalization(self):
        vectors = np.array([[1, 1], [1, 0]], dtype=float)
        Q = gram_schmidt(vectors)
        # 驗證正交性
        assert abs(np.dot(Q[0], Q[1])) < 1e-10
        # 驗證單位向量
        for q in Q:
            assert np.linalg.norm(q) == pytest.approx(1.0)

    def test_3d_orthogonalization(self):
        vectors = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        Q = gram_schmidt(vectors)
        # 驗證 Q^T Q ≈ I
        assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)

    def test_dependent_vectors_raise(self):
        vectors = np.array([[1, 2], [2, 4]], dtype=float)
        with pytest.raises(ValueError):
            gram_schmidt(vectors)

    def test_matches_qr(self):
        """Gram-Schmidt 等價於 QR 分解的 Q"""
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        Q_gs = gram_schmidt(A)
        Q_np, _ = np.linalg.qr(A.T)
        # Q 的列空間相同（符號可能不同）
        for i in range(3):
            dot = abs(np.dot(Q_gs[i], Q_np[:, i]))
            assert dot == pytest.approx(1.0, abs=1e-10)


class TestProjection:
    """4.4 投影"""

    def test_projection_onto_line(self):
        """投影到 x 軸"""
        v = np.array([3, 4])
        basis = np.array([[1, 0]])
        proj = projection_onto_subspace(v, basis)
        assert_allclose(proj, [3, 0])

    def test_projection_idempotent(self):
        """投影的冪等性: P(P(v)) = P(v)"""
        v = np.array([1, 2, 3])
        basis = np.array([[1, 1, 0], [0, 1, 1]])
        Q = gram_schmidt(basis)
        proj1 = projection_onto_subspace(v, Q)
        proj2 = projection_onto_subspace(proj1, Q)
        assert_allclose(proj1, proj2, atol=1e-10)

    def test_remove_gravity(self):
        """加速度計去重力：投影到非重力平面"""
        gravity = np.array([0, 0, 9.81])
        total_accel = np.array([0.5, -0.3, 9.81 + 0.1])
        g_unit = gravity / np.linalg.norm(gravity)
        # 投影到重力方向
        g_proj = np.dot(total_accel, g_unit) * g_unit
        # 去重力後的加速度
        accel_no_gravity = total_accel - g_proj
        assert abs(accel_no_gravity[2]) < 0.2  # z 分量大幅減小


class TestRankAndNullSpace:
    """4.1–4.3 秩、線性獨立、零空間"""

    def test_rank_full(self):
        A = np.array([[1, 0], [0, 1]])
        assert np.linalg.matrix_rank(A) == 2

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]])
        assert np.linalg.matrix_rank(A) == 1

    def test_rank_nullity_theorem(self):
        """秩-零化度定理: rank(A) + nullity(A) = n"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rank = np.linalg.matrix_rank(A)
        null_space = sla.null_space(A)
        nullity = null_space.shape[1]
        assert rank + nullity == A.shape[1]

    def test_null_space_verification(self):
        """Ax = 0 for x in null space"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ns = sla.null_space(A)
        if ns.shape[1] > 0:
            for i in range(ns.shape[1]):
                assert_allclose(A @ ns[:, i], np.zeros(3), atol=1e-10)

    def test_truss_rigid_body_modes(self):
        """未約束桁架的零空間 = 剛體運動"""
        # 1D 簡化：兩個自由節點的剛度矩陣
        k = 1.0
        K = np.array([[k, -k], [-k, k]])
        ns = sla.null_space(K)
        # 零空間維度 = 1（平移模態）
        assert ns.shape[1] == 1
        # 零空間向量代表均勻平移 [1, 1]^T
        assert_allclose(np.abs(ns[:, 0] / ns[0, 0]), [1, 1], atol=1e-10)

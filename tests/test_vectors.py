"""Module 1 & 2 — 向量與矩陣運算測試"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.linalg_utils import (
    vector_add, scalar_multiply, dot_product, cross_product,
    vector_norm, unit_vector, rotation_matrix_2d, rotation_matrix_3d,
    homogeneous_transform_2d, determinant_2x2, determinant_3x3,
    inverse_2x2,
)


class TestVectorOperations:
    """1.1–1.5 向量基礎運算"""

    def test_vector_add(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = vector_add(a, b)
        assert_allclose(result, [5, 7, 9])

    def test_vector_add_dimension_mismatch(self):
        with pytest.raises(ValueError):
            vector_add([1, 2], [1, 2, 3])

    def test_scalar_multiply(self):
        v = [1, 2, 3]
        assert_allclose(scalar_multiply(2, v), [2, 4, 6])
        assert_allclose(scalar_multiply(-1, v), [-1, -2, -3])

    def test_dot_product(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert dot_product(a, b) == pytest.approx(32.0)

    def test_dot_product_orthogonal(self):
        """正交向量內積為零"""
        a = [1, 0]
        b = [0, 1]
        assert dot_product(a, b) == pytest.approx(0.0)

    def test_cross_product(self):
        a = [1, 0, 0]
        b = [0, 1, 0]
        result = cross_product(a, b)
        assert_allclose(result, [0, 0, 1])
        # 驗證與 NumPy 一致
        assert_allclose(result, np.cross(a, b))

    def test_cross_product_anticommutative(self):
        """a × b = -(b × a)"""
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert_allclose(cross_product(a, b), -cross_product(b, a))

    def test_cross_product_parallel(self):
        """平行向量外積為零向量"""
        a = [1, 2, 3]
        b = [2, 4, 6]
        assert_allclose(cross_product(a, b), [0, 0, 0], atol=1e-15)

    def test_cross_product_2d_raises(self):
        with pytest.raises(ValueError):
            cross_product([1, 2], [3, 4])

    def test_vector_norm_l2(self):
        v = [3, 4]
        assert vector_norm(v, 2) == pytest.approx(5.0)

    def test_vector_norm_l1(self):
        v = [3, -4]
        assert vector_norm(v, 1) == pytest.approx(7.0)

    def test_vector_norm_linf(self):
        v = [3, -4, 2]
        assert vector_norm(v, np.inf) == pytest.approx(4.0)

    def test_vector_norm_matches_numpy(self):
        v = [1.5, -2.3, 4.1]
        for p in [1, 2, 3]:
            assert vector_norm(v, p) == pytest.approx(np.linalg.norm(v, ord=p))

    def test_unit_vector(self):
        v = [3, 4]
        u = unit_vector(v)
        assert_allclose(np.linalg.norm(u), 1.0)
        assert_allclose(u, [0.6, 0.8])

    def test_unit_vector_zero_raises(self):
        with pytest.raises(ValueError):
            unit_vector([0, 0, 0])


class TestWorkPhysics:
    """1.3 功的計算 W = F · d"""

    def test_work_parallel(self):
        """力與位移同方向"""
        F = [10, 0]
        d = [5, 0]
        W = dot_product(F, d)
        assert W == pytest.approx(50.0)

    def test_work_perpendicular(self):
        """力與位移垂直，做功為零（磁力）"""
        F = [0, 10]
        d = [5, 0]
        W = dot_product(F, d)
        assert W == pytest.approx(0.0)

    def test_work_at_angle(self):
        """力與位移有夾角"""
        F = np.array([10, 0])
        d = np.array([5 * np.cos(np.pi/3), 5 * np.sin(np.pi/3)])
        W = dot_product(F, d)
        assert W == pytest.approx(10 * 5 * np.cos(np.pi/3))


class TestTorque:
    """1.4 力矩 τ = r × F"""

    def test_wrench_torque(self):
        """扳手鎖螺絲"""
        r = [0.3, 0, 0]  # 30cm 扳手
        F = [0, 50, 0]   # 50N 垂直施力
        tau = cross_product(r, F)
        assert tau[2] == pytest.approx(15.0)  # |τ| = 0.3 * 50 = 15 N·m

    def test_torque_magnitude_equals_area(self):
        """|a × b| = |a||b|sinθ = 平行四邊形面積"""
        a = np.array([3, 0, 0])
        b = np.array([2 * np.cos(np.pi/6), 2 * np.sin(np.pi/6), 0])
        tau = cross_product(a, b)
        assert np.linalg.norm(tau) == pytest.approx(3 * 2 * np.sin(np.pi/6), abs=1e-10)


class TestMatrixOperations:
    """Module 2 — 矩陣運算"""

    def test_rotation_2d(self):
        """旋轉 90° 將 (1,0) 映射到 (0,1)"""
        R = rotation_matrix_2d(np.pi / 2)
        v = np.array([1, 0])
        assert_allclose(R @ v, [0, 1], atol=1e-10)

    def test_rotation_2d_composition(self):
        """R(θ₁)·R(θ₂) = R(θ₁+θ₂)"""
        theta1, theta2 = np.pi / 6, np.pi / 4
        R1 = rotation_matrix_2d(theta1)
        R2 = rotation_matrix_2d(theta2)
        R_sum = rotation_matrix_2d(theta1 + theta2)
        assert_allclose(R1 @ R2, R_sum, atol=1e-10)

    def test_rotation_is_orthogonal(self):
        """旋轉矩陣是正交矩陣: R^T R = I, det(R) = 1"""
        for theta in [0, np.pi/4, np.pi/3, np.pi, 2.5]:
            R = rotation_matrix_2d(theta)
            assert_allclose(R.T @ R, np.eye(2), atol=1e-10)
            assert np.linalg.det(R) == pytest.approx(1.0)

    def test_rotation_3d_orthogonal(self):
        for axis in ['x', 'y', 'z']:
            R = rotation_matrix_3d(axis, np.pi / 5)
            assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            assert np.linalg.det(R) == pytest.approx(1.0)

    def test_homogeneous_transform(self):
        """齊次變換：旋轉90° + 平移(1,2)"""
        H = homogeneous_transform_2d(np.pi / 2, 1.0, 2.0)
        p = np.array([1, 0, 1])  # 齊次座標
        result = H @ p
        assert_allclose(result[:2], [1, 3], atol=1e-10)  # (0,1) + (1,2)

    def test_determinant_2x2(self):
        A = [[3, 8], [4, 6]]
        assert determinant_2x2(A) == pytest.approx(np.linalg.det(A))

    def test_determinant_3x3(self):
        A = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
        assert determinant_3x3(A) == pytest.approx(np.linalg.det(A))

    def test_inverse_2x2(self):
        A = np.array([[4, 7], [2, 6]])
        A_inv = inverse_2x2(A)
        assert_allclose(A @ A_inv, np.eye(2), atol=1e-10)
        assert_allclose(A_inv, np.linalg.inv(A), atol=1e-10)

    def test_inverse_singular_raises(self):
        with pytest.raises(ValueError):
            inverse_2x2([[1, 2], [2, 4]])

    def test_det_zero_coplanar_forces(self):
        """三個力共面 → 行列式為零"""
        f1 = [1, 0, 0]
        f2 = [0, 1, 0]
        f3 = [1, 1, 0]  # = f1 + f2，共面
        A = np.array([f1, f2, f3])
        assert abs(np.linalg.det(A)) < 1e-10

"""Module 5 — 矩陣分解測試"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.linalg_utils import (
    lu_decomposition, qr_decomposition, cholesky_decomposition,
    least_squares_normal, ridge_regression, pca_eig, pca_svd,
)


class TestLU:
    """5.1 LU 分解"""

    def test_lu_basic(self):
        A = np.array([[2, 3], [4, 7]], dtype=float)
        L, U = lu_decomposition(A)
        assert_allclose(L @ U, A, atol=1e-10)

    def test_lu_3x3(self):
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
        L, U = lu_decomposition(A)
        assert_allclose(L @ U, A, atol=1e-10)
        # L 是下三角
        assert_allclose(np.triu(L, 1), 0)
        # U 是上三角
        assert_allclose(np.tril(U, -1), 0)

    def test_lu_solve_multiple_rhs(self):
        """LU 分解後解多組荷載"""
        A = np.array([[4, 3], [6, 3]], dtype=float)
        L, U = lu_decomposition(A)

        for b in [[1, 0], [0, 1], [5, 3]]:
            b = np.array(b, dtype=float)
            # Ly = b (forward substitution)
            y = np.linalg.solve(L, b)
            # Ux = y (back substitution)
            x = np.linalg.solve(U, y)
            assert_allclose(A @ x, b, atol=1e-10)


class TestQR:
    """5.2 QR 分解"""

    def test_qr_basic(self):
        A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        Q, R = qr_decomposition(A)
        assert_allclose(Q @ R, A, atol=1e-10)

    def test_qr_orthogonality(self):
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        Q, R = qr_decomposition(A)
        assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_qr_matches_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.normal(size=(5, 3))
        Q, R = qr_decomposition(A)
        Q_np, R_np = np.linalg.qr(A)
        # 重組應一致
        assert_allclose(Q @ R, Q_np @ R_np, atol=1e-10)

    def test_qr_rotation_correction(self):
        """噪聲旋轉矩陣 QR 正交化"""
        R_true = np.array([[np.cos(0.3), -np.sin(0.3)],
                           [np.sin(0.3),  np.cos(0.3)]])
        # 加噪聲
        R_noisy = R_true + np.random.default_rng(42).normal(0, 0.01, (2, 2))
        Q, _ = qr_decomposition(R_noisy)
        # Q 應是正交的
        assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)


class TestCholesky:
    """5.3 Cholesky 分解"""

    def test_cholesky_basic(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky_decomposition(A)
        assert_allclose(L @ L.T, A, atol=1e-10)

    def test_cholesky_matches_numpy(self):
        A = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype=float)
        L = cholesky_decomposition(A)
        L_np = np.linalg.cholesky(A)
        assert_allclose(L, L_np, atol=1e-10)

    def test_cholesky_heat_conduction(self):
        """穩態熱傳導矩陣（對稱正定）"""
        n = 5
        K = np.diag(2 * np.ones(n)) + np.diag(-np.ones(n-1), 1) + np.diag(-np.ones(n-1), -1)
        L = cholesky_decomposition(K)
        assert_allclose(L @ L.T, K, atol=1e-10)

    def test_not_positive_definite_raises(self):
        A = np.array([[1, 2], [2, 1]], dtype=float)  # 特徵值 3, -1
        with pytest.raises(ValueError):
            cholesky_decomposition(A)


class TestSVD:
    """5.4 SVD"""

    def test_svd_reconstruction(self):
        """A = UΣV^T 重組誤差 < 1e-12"""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        A_reconstructed = U @ np.diag(S) @ Vt
        assert_allclose(A_reconstructed, A, atol=1e-12)

    def test_svd_image_compression(self):
        """SVD 影像壓縮：保留 k 個奇異值"""
        rng = np.random.default_rng(42)
        # 低秩影像 + 噪聲
        true_rank = 3
        A = rng.normal(size=(50, true_rank)) @ rng.normal(size=(true_rank, 40))
        A += rng.normal(0, 0.01, A.shape)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # 前 3 個奇異值應遠大於其餘
        assert S[true_rank - 1] / S[true_rank] > 10

    def test_svd_manipulability(self):
        """機器手臂 Jacobian 的 SVD → 可操作性"""
        # 2-DOF 機器手臂
        theta1, theta2 = np.pi/4, np.pi/6
        L1, L2 = 1.0, 0.8
        J = np.array([
            [-L1*np.sin(theta1) - L2*np.sin(theta1+theta2),
             -L2*np.sin(theta1+theta2)],
            [L1*np.cos(theta1) + L2*np.cos(theta1+theta2),
             L2*np.cos(theta1+theta2)],
        ])
        _, S, _ = np.linalg.svd(J)
        # 奇異值都應 > 0（非奇異位形）
        assert all(s > 0.01 for s in S)
        # 可操作性 = σ_min / σ_max
        manipulability = S[-1] / S[0]
        assert 0 < manipulability <= 1


class TestLeastSquares:
    """Module 7 — 最小平方"""

    def test_least_squares_overdetermined(self):
        """超定系統最小平方解"""
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        b = np.array([1, 2, 2], dtype=float)
        x = least_squares_normal(A, b)
        x_np, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x, x_np, atol=1e-10)

    def test_spring_constant_fit(self):
        """彈簧常數擬合 F = kx"""
        true_k = 150.0
        x = np.linspace(0, 0.1, 20)
        F = true_k * x + np.random.default_rng(42).normal(0, 0.5, 20)
        A = x.reshape(-1, 1)
        k_fit = least_squares_normal(A, F)
        assert k_fit[0] == pytest.approx(true_k, abs=5.0)

    def test_ridge_reduces_norm(self):
        """Ridge 正規化使解的範數更小"""
        rng = np.random.default_rng(42)
        A = rng.normal(size=(20, 5))
        b = rng.normal(size=20)
        x_ols = least_squares_normal(A, b)
        x_ridge = ridge_regression(A, b, lam=1.0)
        assert np.linalg.norm(x_ridge) < np.linalg.norm(x_ols)


class TestPCA:
    """Module 8 — PCA"""

    def test_pca_eig_svd_consistency(self):
        """特徵分解法與 SVD 法的主成分方向一致"""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        X[:, 0] += 3 * X[:, 1]  # 人為相關性

        comp_eig, var_eig, _ = pca_eig(X, n_components=3)
        comp_svd, var_svd, _ = pca_svd(X, n_components=3)

        # 方差應一致
        assert_allclose(var_eig, var_svd, atol=1e-8)

        # 主成分方向一致（允許符號翻轉）
        for i in range(3):
            dot = abs(np.dot(comp_eig[i], comp_svd[i]))
            assert dot == pytest.approx(1.0, abs=1e-8)

    def test_pca_explained_variance(self):
        """解釋變異量比例總和 <= 1"""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 4))
        _, var, _ = pca_eig(X)
        ratio = var / var.sum()
        assert ratio.sum() == pytest.approx(1.0)
        assert all(r >= 0 for r in ratio)

    def test_pca_matches_sklearn(self):
        """與 sklearn PCA 結果比對"""
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))

        pca_sk = PCA(n_components=3)
        pca_sk.fit(X)

        comp, var, _ = pca_eig(X, n_components=3)

        assert_allclose(var, pca_sk.explained_variance_, atol=1e-8)
        for i in range(3):
            dot = abs(np.dot(comp[i], pca_sk.components_[i]))
            assert dot == pytest.approx(1.0, abs=1e-6)

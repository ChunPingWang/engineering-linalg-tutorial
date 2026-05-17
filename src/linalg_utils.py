"""
手動實作核心線性代數算法（教學用）
Manual implementations of core linear algebra algorithms for educational purposes.
"""

import numpy as np


# ============================================================
# Module 1: 向量運算
# ============================================================

def vector_add(a, b):
    """向量加法 — 手動實作"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"向量維度不一致: {a.shape} vs {b.shape}")
    return a + b


def scalar_multiply(scalar, v):
    """純量乘法"""
    return float(scalar) * np.asarray(v, dtype=float)


def dot_product(a, b):
    """內積 — 手動實作 (逐元素相乘再求和)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"向量維度不一致: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def cross_product(a, b):
    """外積 — 手動實作 (僅限 3D)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("外積僅適用於 3D 向量")
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def vector_norm(v, p=2):
    """Lp 範數 — 手動實作"""
    v = np.asarray(v, dtype=float)
    if p == np.inf:
        return float(np.max(np.abs(v)))
    return float(np.sum(np.abs(v) ** p) ** (1.0 / p))


def unit_vector(v):
    """單位向量"""
    v = np.asarray(v, dtype=float)
    n = vector_norm(v)
    if n < 1e-15:
        raise ValueError("零向量無法正規化")
    return v / n


# ============================================================
# Module 2: 矩陣運算
# ============================================================

def rotation_matrix_2d(theta):
    """2D 旋轉矩陣 R(θ)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_matrix_3d(axis, theta):
    """3D 旋轉矩陣 (繞指定軸 'x', 'y', 'z')"""
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"未知軸: {axis}, 請使用 'x', 'y', 或 'z'")


def homogeneous_transform_2d(theta, tx, ty):
    """2D 齊次變換矩陣 (旋轉 + 平移)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1],
    ])


def determinant_2x2(A):
    """2x2 行列式 — 手動實作"""
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("僅適用於 2x2 矩陣")
    return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


def determinant_3x3(A):
    """3x3 行列式 — 手動展開"""
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("僅適用於 3x3 矩陣")
    return float(
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


def inverse_2x2(A):
    """2x2 逆矩陣 — 手動實作"""
    A = np.asarray(A, dtype=float)
    det = determinant_2x2(A)
    if abs(det) < 1e-15:
        raise ValueError("矩陣為奇異矩陣，無法求逆")
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


# ============================================================
# Module 3: 高斯消去法
# ============================================================

def gaussian_elimination(A, b):
    """高斯消去法 (含 Partial Pivoting)

    求解 Ax = b

    Parameters
    ----------
    A : array-like, shape (n, n)
    b : array-like, shape (n,)

    Returns
    -------
    x : ndarray, shape (n,)
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(b)

    # 前向消去 (Forward Elimination) with Partial Pivoting
    for k in range(n):
        # 找最大絕對值的行作為樞軸
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[max_row, k]) < 1e-15:
            raise ValueError("矩陣奇異，無法求解")
        # 交換行
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        # 消去
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # 回代 (Back Substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x


# ============================================================
# Module 4: 向量空間工具
# ============================================================

def gram_schmidt(vectors):
    """Gram-Schmidt 正交化

    Parameters
    ----------
    vectors : array-like, shape (k, n)
        k 個 n 維向量（按行排列）

    Returns
    -------
    Q : ndarray, shape (k, n)
        正交化後的單位向量
    """
    V = np.asarray(vectors, dtype=float).copy()
    k, n = V.shape
    Q = np.zeros_like(V)
    for i in range(k):
        q = V[i].copy()
        for j in range(i):
            q -= np.dot(Q[j], V[i]) * Q[j]
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError(f"第 {i} 個向量線性相依，無法正交化")
        Q[i] = q / norm
    return Q


def projection_onto_subspace(v, basis):
    """將向量 v 投影到由 basis 張成的子空間

    Parameters
    ----------
    v : array-like, shape (n,)
    basis : array-like, shape (k, n)
        子空間的正交基（按行排列）

    Returns
    -------
    proj : ndarray, shape (n,)
    """
    v = np.asarray(v, dtype=float)
    basis = np.asarray(basis, dtype=float)
    proj = np.zeros_like(v)
    for b in basis:
        proj += np.dot(b, v) / np.dot(b, b) * b
    return proj


# ============================================================
# Module 5: 矩陣分解
# ============================================================

def lu_decomposition(A):
    """LU 分解（無 Pivoting）— 教學用

    Returns
    -------
    L, U : ndarray
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n):
        if abs(U[k, k]) < 1e-15:
            raise ValueError(f"樞軸元素為零 (位置 {k})，需要 Pivoting")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U


def qr_decomposition(A):
    """QR 分解 — 基於 Gram-Schmidt

    Returns
    -------
    Q : ndarray, shape (m, n) — 正交矩陣
    R : ndarray, shape (n, n) — 上三角矩陣
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-12:
            raise ValueError(f"第 {j} 列線性相依")
        Q[:, j] = v / R[j, j]
    return Q, R


def cholesky_decomposition(A):
    """Cholesky 分解 — A = LL^T（對稱正定矩陣）

    Returns
    -------
    L : ndarray — 下三角矩陣
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError("矩陣不是正定矩陣")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


# ============================================================
# Module 7: 最小平方
# ============================================================

def least_squares_normal(A, b):
    """Normal Equation 求最小平方解: x = (A^T A)^{-1} A^T b"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    ATA = A.T @ A
    ATb = A.T @ b
    return np.linalg.solve(ATA, ATb)


def ridge_regression(A, b, lam):
    """Ridge Regression: x = (A^T A + λI)^{-1} A^T b"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


# ============================================================
# Module 8: PCA
# ============================================================

def pca_eig(X, n_components=None):
    """PCA — 特徵分解法

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features) — 原始資料（未中心化）
    n_components : int, optional

    Returns
    -------
    components : ndarray, shape (n_components, n_features) — 主成分方向
    explained_variance : ndarray — 各主成分的方差
    X_transformed : ndarray — 投影後資料
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 從大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    components = eigenvectors.T
    X_transformed = X_centered @ eigenvectors
    return components, eigenvalues, X_transformed


def pca_svd(X, n_components=None):
    """PCA — SVD 法

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int, optional

    Returns
    -------
    components : ndarray — 主成分方向
    explained_variance : ndarray
    X_transformed : ndarray
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = S ** 2 / (X.shape[0] - 1)

    if n_components is not None:
        Vt = Vt[:n_components]
        explained_variance = explained_variance[:n_components]

    components = Vt
    X_transformed = X_centered @ Vt.T
    return components, explained_variance, X_transformed

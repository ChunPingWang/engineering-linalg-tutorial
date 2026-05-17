"""
視覺化工具 — 向量場、軌跡、相平面
Visualization utilities for the linear algebra tutorial.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


def set_style():
    """設定全局繪圖風格"""
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
    })


def plot_vectors_2d(vectors, labels=None, colors=None, origin=None, ax=None,
                    title="2D 向量圖"):
    """繪製 2D 向量（箭頭圖）

    Parameters
    ----------
    vectors : list of array-like, each shape (2,)
    labels : list of str, optional
    colors : list of str, optional
    origin : array-like, shape (2,), optional — 共同起點
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if origin is None:
        origin = np.zeros(2)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))

    all_coords = [origin]
    for i, v in enumerate(vectors):
        v = np.asarray(v)
        color = colors[i] if i < len(colors) else 'blue'
        label = labels[i] if labels and i < len(labels) else None
        ax.annotate('', xy=origin + v, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if label:
            mid = origin + v * 0.5
            ax.text(mid[0], mid[1], f' {label}', fontsize=11,
                    color=color, fontweight='bold')
        all_coords.append(origin + v)

    all_coords = np.array(all_coords)
    margin = max(np.ptp(all_coords, axis=0)) * 0.2 + 0.5
    ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_trajectory_2d(x, y, vx=None, vy=None, ax=None, title="2D 軌跡",
                       arrow_step=10):
    """繪製 2D 軌跡並可選附加速度箭頭"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, 'b-', lw=2, label='軌跡')
    ax.plot(x[0], y[0], 'go', ms=8, label='起點')

    if vx is not None and vy is not None:
        scale = max(np.max(np.abs(vx)), np.max(np.abs(vy))) * 0.1
        for i in range(0, len(x), arrow_step):
            ax.annotate('', xy=(x[i] + vx[i]*scale, y[i] + vy[i]*scale),
                        xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_matrix_heatmap(A, title="矩陣熱力圖", fmt=".2f", ax=None):
    """矩陣視覺化（熱力圖）"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    A = np.asarray(A)
    im = ax.imshow(A, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i,j]:{fmt}}', ha='center', va='center', fontsize=9)

    ax.set_title(title)
    return ax


def plot_phase_portrait(A, xlim=(-5, 5), ylim=(-5, 5), n_trajectories=20,
                        t_max=5.0, ax=None, title="相平面圖"):
    """繪製 2D 線性系統 dx/dt = Ax 的相平面"""
    from scipy.integrate import solve_ivp

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # 向量場
    xg = np.linspace(xlim[0], xlim[1], 20)
    yg = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(xg, yg)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    ax.streamplot(X, Y, U, V, color='lightblue', density=1, linewidth=0.5)

    # 軌跡
    angles = np.linspace(0, 2 * np.pi, n_trajectories, endpoint=False)
    r = max(abs(xlim[1]), abs(ylim[1])) * 0.8
    for angle in angles:
        x0 = [r * np.cos(angle), r * np.sin(angle)]
        sol = solve_ivp(lambda t, s: A @ s, [0, t_max], x0,
                        max_step=0.05, dense_output=True)
        ax.plot(sol.y[0], sol.y[1], 'b-', lw=0.8, alpha=0.7)
        if len(sol.y[0]) > 1:
            ax.annotate('', xy=(sol.y[0][-1], sol.y[1][-1]),
                        xytext=(sol.y[0][-2], sol.y[1][-2]),
                        arrowprops=dict(arrowstyle='->', color='blue'))

    eigenvalues = np.linalg.eigvals(A)
    eig_str = ', '.join([f'{e:.2f}' for e in eigenvalues])
    ax.set_title(f'{title}\n特徵值: {eig_str}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return ax


def plot_ellipsoid_2d(center, A_matrix, ax=None, title="2D 橢球",
                      color='blue', alpha=0.3):
    """繪製 2D 橢圓 x^T A x = 1"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    eigenvalues, eigenvectors = np.linalg.eigh(A_matrix)
    theta = np.linspace(0, 2 * np.pi, 200)
    # 在特徵基底上畫單位圓，縮放後轉回原坐標
    radii = 1.0 / np.sqrt(eigenvalues)
    circle = np.array([radii[0] * np.cos(theta), radii[1] * np.sin(theta)])
    ellipse = eigenvectors @ circle + np.array(center)[:, None]

    ax.fill(ellipse[0], ellipse[1], alpha=alpha, color=color)
    ax.plot(ellipse[0], ellipse[1], color=color, lw=2)

    # 主軸
    for i in range(2):
        direction = eigenvectors[:, i] * radii[i]
        ax.annotate('', xy=center + direction, xytext=center,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_temperature_field(X, Y, T, title="溫度場", ax=None):
    """繪製 2D 溫度場等值線圖"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    cf = ax.contourf(X, Y, T, levels=30, cmap='hot')
    plt.colorbar(cf, ax=ax, label='Temperature (°C)')
    ax.contour(X, Y, T, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_mode_shapes(mode_shapes, frequencies, n_modes=None):
    """繪製振型圖"""
    n_dof = mode_shapes.shape[0]
    if n_modes is None:
        n_modes = min(mode_shapes.shape[1], 4)

    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4))
    if n_modes == 1:
        axes = [axes]

    x = np.arange(n_dof)
    for i in range(n_modes):
        ax = axes[i]
        shape = mode_shapes[:, i]
        shape = shape / np.max(np.abs(shape))  # 正規化
        ax.bar(x, shape, color='steelblue', alpha=0.7)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'Mode {i+1}\nf = {frequencies[i]:.2f} Hz')
        ax.set_xlabel('DOF')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_svd_compression(original, reconstructed_list, k_values, title="SVD 壓縮"):
    """比較不同 k 值的 SVD 影像壓縮結果"""
    n = len(k_values) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'原始 ({original.shape[0]}x{original.shape[1]})')
    axes[0].axis('off')

    for i, (recon, k) in enumerate(zip(reconstructed_list, k_values)):
        axes[i + 1].imshow(recon, cmap='gray')
        error = np.linalg.norm(original - recon) / np.linalg.norm(original)
        axes[i + 1].set_title(f'k={k}, 誤差={error:.4f}')
        axes[i + 1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    return fig, axes

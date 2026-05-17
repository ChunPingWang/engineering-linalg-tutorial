"""
物理系統模型 — 彈簧、熱傳、流場、控制
Physics system models for engineering linear algebra tutorial.
"""

import numpy as np
from scipy import linalg as sla
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ============================================================
# 運動學模型 (Module 1)
# ============================================================

def projectile_trajectory(v0, theta, g=9.81, dt=0.01, t_max=None):
    """2D 拋體運動軌跡

    Parameters
    ----------
    v0 : float — 初速 (m/s)
    theta : float — 仰角 (rad)
    g : float — 重力加速度
    dt : float — 時間步長
    t_max : float, optional — 若未指定則飛行至落地

    Returns
    -------
    t, x, y, vx, vy : ndarray — 時間、位置、速度分量
    """
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    if t_max is None:
        t_max = 2 * vy0 / g + 0.01

    t = np.arange(0, t_max, dt)
    x = vx0 * t
    y = vy0 * t - 0.5 * g * t ** 2
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t

    # 截斷到 y >= 0
    mask = y >= 0
    return t[mask], x[mask], y[mask], vx[mask], vy[mask]


# ============================================================
# 靜力學模型 (Module 3)
# ============================================================

def truss_2d(nodes, elements, supports, loads):
    """2D 桁架分析 — 節點法

    Parameters
    ----------
    nodes : dict {node_id: (x, y)}
    elements : list of (node_i, node_j) — 桿件連接
    supports : dict {node_id: (fix_x, fix_y)} — True=固定
    loads : dict {node_id: (Fx, Fy)} — 外力

    Returns
    -------
    displacements : dict {node_id: (ux, uy)}
    forces : dict — 各桿件軸力
    """
    node_ids = sorted(nodes.keys())
    n_nodes = len(node_ids)
    n_dof = 2 * n_nodes
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    K_global = np.zeros((n_dof, n_dof))
    element_props = {}

    for elem_id, (ni, nj) in enumerate(elements):
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
        c = (xj - xi) / L
        s = (yj - yi) / L

        # 假設 EA = 1（可擴展）
        EA_over_L = 1.0 / L

        k_local = EA_over_L * np.array([
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c,  c*s],
            [-c*s, -s*s, c*s,  s*s],
        ])

        dofs = [2*node_idx[ni], 2*node_idx[ni]+1,
                2*node_idx[nj], 2*node_idx[nj]+1]

        for a in range(4):
            for b in range(4):
                K_global[dofs[a], dofs[b]] += k_local[a, b]

        element_props[elem_id] = {'L': L, 'c': c, 's': s, 'dofs': dofs}

    # 荷載向量
    F = np.zeros(n_dof)
    for nid, (fx, fy) in loads.items():
        idx = node_idx[nid]
        F[2*idx] = fx
        F[2*idx+1] = fy

    # 邊界條件
    free_dofs = list(range(n_dof))
    for nid, (fix_x, fix_y) in supports.items():
        idx = node_idx[nid]
        if fix_x:
            free_dofs.remove(2*idx)
        if fix_y:
            free_dofs.remove(2*idx+1)

    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    U = np.zeros(n_dof)
    U[free_dofs] = np.linalg.solve(K_ff, F_f)

    displacements = {nid: (U[2*node_idx[nid]], U[2*node_idx[nid]+1])
                     for nid in node_ids}

    forces = {}
    for elem_id, (ni, nj) in enumerate(elements):
        props = element_props[elem_id]
        dofs = props['dofs']
        u_elem = U[dofs]
        c, s, L = props['c'], props['s'], props['L']
        axial_force = (1.0 / L) * np.dot([-c, -s, c, s], u_elem)
        forces[elem_id] = axial_force

    return displacements, forces


# ============================================================
# 彈簧-質量系統 (Module 6)
# ============================================================

def spring_mass_system(masses, springs):
    """多自由度彈簧-質量系統

    Parameters
    ----------
    masses : list of float — 各質量 (kg)
    springs : list of (i, j, k) — 彈簧連接 (節點i, 節點j, 彈簧常數k)
        節點 -1 表示固定牆

    Returns
    -------
    M : ndarray — 質量矩陣
    K : ndarray — 剛度矩陣
    frequencies : ndarray — 固有頻率 (Hz)
    mode_shapes : ndarray — 振型 (列向量)
    """
    n = len(masses)
    M = np.diag(masses)
    K = np.zeros((n, n))

    for (i, j, k_val) in springs:
        if i >= 0 and j >= 0:
            K[i, i] += k_val
            K[j, j] += k_val
            K[i, j] -= k_val
            K[j, i] -= k_val
        elif i == -1:  # 固定牆-j
            K[j, j] += k_val
        elif j == -1:  # i-固定牆
            K[i, i] += k_val

    eigenvalues, eigenvectors = sla.eigh(K, M)
    frequencies = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    return M, K, frequencies, eigenvectors


def simulate_spring_mass(M, K, x0, v0, t_span, C=None, F_ext=None, dt=0.001):
    """模擬彈簧-質量系統的時域響應

    Parameters
    ----------
    M, K : ndarray — 質量、剛度矩陣
    x0, v0 : ndarray — 初始位移、初始速度
    t_span : tuple (t0, tf)
    C : ndarray, optional — 阻尼矩陣
    F_ext : callable(t) -> ndarray, optional — 外力函數
    dt : float — 輸出時間步長

    Returns
    -------
    t : ndarray — 時間
    x : ndarray — 位移 (n_dof, n_time)
    """
    n = len(x0)
    M_inv = np.linalg.inv(M)
    if C is None:
        C = np.zeros((n, n))
    if F_ext is None:
        F_ext = lambda t: np.zeros(n)

    def equations(t, state):
        x = state[:n]
        v = state[n:]
        a = M_inv @ (F_ext(t) - C @ v - K @ x)
        return np.concatenate([v, a])

    state0 = np.concatenate([x0, v0])
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(equations, t_span, state0, t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[:n]


# ============================================================
# 熱傳導模型 (Module 5, 9)
# ============================================================

def heat_conduction_1d(n_nodes, T_left, T_right, q_internal=None, k=1.0, L=1.0):
    """一維穩態熱傳導

    Parameters
    ----------
    n_nodes : int — 內部節點數
    T_left, T_right : float — 左右邊界溫度
    q_internal : ndarray, optional — 內部熱源
    k : float — 導熱係數
    L : float — 總長度

    Returns
    -------
    x : ndarray — 節點位置 (含邊界)
    T : ndarray — 溫度分佈 (含邊界)
    K_matrix : ndarray — 導熱矩陣
    """
    dx = L / (n_nodes + 1)
    # 三對角矩陣
    main_diag = 2.0 * k / dx ** 2 * np.ones(n_nodes)
    off_diag = -k / dx ** 2 * np.ones(n_nodes - 1)
    K_matrix = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # 右手邊
    rhs = np.zeros(n_nodes)
    if q_internal is not None:
        rhs += q_internal
    rhs[0] += k / dx ** 2 * T_left
    rhs[-1] += k / dx ** 2 * T_right

    T_internal = np.linalg.solve(K_matrix, rhs)
    x = np.linspace(0, L, n_nodes + 2)
    T = np.concatenate([[T_left], T_internal, [T_right]])
    return x, T, K_matrix


def heat_conduction_2d(nx, ny, T_boundary, k=1.0, Lx=1.0, Ly=1.0):
    """二維穩態熱傳導（泊松方程）

    Parameters
    ----------
    nx, ny : int — x, y 方向內部節點數
    T_boundary : dict — {'top': T, 'bottom': T, 'left': T, 'right': T}
    k : float — 導熱係數

    Returns
    -------
    X, Y : ndarray — 網格座標 (含邊界)
    T : ndarray — 溫度場 (含邊界)
    """
    dx = Lx / (nx + 1)
    dy = Ly / (ny + 1)
    N = nx * ny

    # 建構稀疏係數矩陣
    main = (2.0/dx**2 + 2.0/dy**2) * np.ones(N)
    off_x = -1.0/dx**2 * np.ones(N - 1)
    off_y = -1.0/dy**2 * np.ones(N - nx)

    # 消除跨行的連接
    for i in range(1, ny):
        off_x[i * nx - 1] = 0.0

    A = diags([main, off_x, off_x, off_y, off_y],
              [0, -1, 1, -nx, nx], format='csr')

    # 右手邊
    rhs = np.zeros(N)
    T_top = T_boundary.get('top', 0)
    T_bot = T_boundary.get('bottom', 0)
    T_left = T_boundary.get('left', 0)
    T_right = T_boundary.get('right', 0)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i == 0:
                rhs[idx] += T_left / dx**2
            if i == nx - 1:
                rhs[idx] += T_right / dx**2
            if j == 0:
                rhs[idx] += T_bot / dy**2
            if j == ny - 1:
                rhs[idx] += T_top / dy**2

    T_internal = spsolve(A, rhs)

    # 組裝完整溫度場
    x = np.linspace(0, Lx, nx + 2)
    y = np.linspace(0, Ly, ny + 2)
    X, Y = np.meshgrid(x, y)
    T = np.zeros((ny + 2, nx + 2))
    T[0, :] = T_bot
    T[-1, :] = T_top
    T[:, 0] = T_left
    T[:, -1] = T_right
    for j in range(ny):
        for i in range(nx):
            T[j + 1, i + 1] = T_internal[j * nx + i]

    return X, Y, T


# ============================================================
# 控制系統模型 (Module 6, 9)
# ============================================================

def inverted_pendulum_linearized(m=1.0, M_cart=5.0, L=2.0, g=9.81):
    """倒擺線性化狀態空間模型

    State: [x, x_dot, theta, theta_dot]

    Returns
    -------
    A, B, C, D : ndarray — 狀態空間矩陣
    """
    denom = M_cart + m
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m * g / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (M_cart + m) * g / (denom * L), 0],
    ])
    B = np.array([
        [0],
        [1 / denom],
        [0],
        [-1 / (denom * L)],
    ])
    C = np.eye(4)
    D = np.zeros((4, 1))
    return A, B, C, D


def controllability_matrix(A, B):
    """可控性矩陣 [B, AB, A^2B, ..., A^{n-1}B]"""
    n = A.shape[0]
    Ctrb = B.copy()
    col = B.copy()
    for _ in range(n - 1):
        col = A @ col
        Ctrb = np.hstack([Ctrb, col])
    return Ctrb


def observability_matrix(A, C):
    """可觀性矩陣 [C; CA; CA^2; ...; CA^{n-1}]"""
    n = A.shape[0]
    Obsv = C.copy()
    row = C.copy()
    for _ in range(n - 1):
        row = row @ A
        Obsv = np.vstack([Obsv, row])
    return Obsv


# ============================================================
# 有限元素法 (Module 9)
# ============================================================

def fem_bar_1d(nodes, elements, E_A_list, boundary, loads):
    """一維桿件有限元素法

    Parameters
    ----------
    nodes : list of float — 節點位置
    elements : list of (i, j) — 元素連接
    E_A_list : list of float — 各元素的 EA 值
    boundary : dict {node_idx: displacement}
    loads : dict {node_idx: force}

    Returns
    -------
    u : ndarray — 節點位移
    stress : list of float — 各元素應力
    K_global : ndarray — 全域剛度矩陣
    """
    n = len(nodes)
    K_global = np.zeros((n, n))

    for elem_id, (i, j) in enumerate(elements):
        L = abs(nodes[j] - nodes[i])
        EA = E_A_list[elem_id]
        ke = EA / L * np.array([[1, -1], [-1, 1]])
        K_global[i, i] += ke[0, 0]
        K_global[i, j] += ke[0, 1]
        K_global[j, i] += ke[1, 0]
        K_global[j, j] += ke[1, 1]

    F = np.zeros(n)
    for idx, force in loads.items():
        F[idx] = force

    free = [i for i in range(n) if i not in boundary]
    K_ff = K_global[np.ix_(free, free)]
    F_f = F[free]
    for i_free in free:
        for bc_node, bc_val in boundary.items():
            F_f[free.index(i_free)] -= K_global[i_free, bc_node] * bc_val

    u = np.zeros(n)
    for bc_node, bc_val in boundary.items():
        u[bc_node] = bc_val
    u[free] = np.linalg.solve(K_ff, F_f)

    stress = []
    for elem_id, (i, j) in enumerate(elements):
        L = abs(nodes[j] - nodes[i])
        strain = (u[j] - u[i]) / L
        stress.append(E_A_list[elem_id] * strain / (E_A_list[elem_id] / 1.0))  # σ = E·ε, here simplified

    # Fix: stress = strain for unit area
    stress = []
    for elem_id, (i, j) in enumerate(elements):
        L = abs(nodes[j] - nodes[i])
        strain = (u[j] - u[i]) / L
        stress.append(strain * E_A_list[elem_id] / L * L)  # σ = EA/A * ε = E*ε, simplified as EA*ε/L * L/A

    # Simplified: just return axial force / unit
    stress = []
    for elem_id, (i, j) in enumerate(elements):
        L = abs(nodes[j] - nodes[i])
        axial_force = E_A_list[elem_id] / L * (u[j] - u[i])
        stress.append(axial_force)

    return u, stress, K_global


# ============================================================
# Markov 鏈 (Module 6)
# ============================================================

def markov_steady_state(P):
    """求 Markov 鏈的穩態分佈

    Parameters
    ----------
    P : ndarray — 轉移矩陣 (行隨機矩陣，各行和為1)

    Returns
    -------
    pi : ndarray — 穩態分佈向量
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # 找最接近 1 的特徵值
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # 歸一化
    return pi

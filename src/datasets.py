"""
模擬感測器、熱電偶、加速度計資料
Simulated sensor data generators for the tutorial.
"""

import numpy as np


def generate_thermocouple_data(n_points=50, true_slope=0.041, true_intercept=-0.2,
                               noise_std=0.005, T_range=(0, 400), seed=42):
    """模擬熱電偶 (溫度, 電壓) 量測資料

    Parameters
    ----------
    n_points : int
    true_slope : float — mV/°C
    true_intercept : float — mV
    noise_std : float — 噪聲標準差 (mV)
    T_range : tuple — 溫度範圍 (°C)
    seed : int

    Returns
    -------
    T : ndarray — 溫度 (°C)
    V : ndarray — 電壓 (mV)
    true_params : dict
    """
    rng = np.random.default_rng(seed)
    T = np.linspace(T_range[0], T_range[1], n_points)
    V = true_slope * T + true_intercept + rng.normal(0, noise_std, n_points)
    return T, V, {'slope': true_slope, 'intercept': true_intercept}


def generate_spring_data(n_points=30, true_k=150.0, noise_std=2.0,
                         x_range=(0, 0.1), seed=42):
    """模擬彈簧力-位移數據 F = kx

    Returns
    -------
    x : ndarray — 位移 (m)
    F : ndarray — 力 (N)
    true_k : float
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_range[0], x_range[1], n_points)
    F = true_k * x + rng.normal(0, noise_std, n_points)
    return x, F, true_k


def generate_imu_data(n_samples=1000, dt=0.01, seed=42):
    """模擬六軸 IMU 資料 (3-axis acc + 3-axis gyro)

    模擬一個靜止但有噪聲的 IMU，加上低頻振動干擾。

    Returns
    -------
    data : ndarray, shape (n_samples, 6) — [ax, ay, az, gx, gy, gz]
    labels : list of str
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    # 加速度計（靜止時只有重力 z 軸 ≈ 9.81）
    acc_noise = 0.05
    ax = rng.normal(0, acc_noise, n_samples) + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    ay = rng.normal(0, acc_noise, n_samples) + 0.2 * np.sin(2 * np.pi * 2.0 * t + 0.5)
    az = 9.81 + rng.normal(0, acc_noise, n_samples) + 0.1 * np.sin(2 * np.pi * 5.0 * t)

    # 陀螺儀
    gyro_noise = 0.01
    gx = rng.normal(0, gyro_noise, n_samples) + 0.05 * np.sin(2 * np.pi * 1.0 * t)
    gy = rng.normal(0, gyro_noise, n_samples) + 0.03 * np.sin(2 * np.pi * 1.0 * t + 0.3)
    gz = rng.normal(0, gyro_noise, n_samples)

    data = np.column_stack([ax, ay, az, gx, gy, gz])
    labels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    return data, labels


def generate_bridge_vibration_data(n_sensors=20, n_samples=500, n_modes=3,
                                   damage_ratio=0.3, seed=42):
    """模擬橋梁振動感測器資料 (含正常與損傷狀態)

    Parameters
    ----------
    n_sensors : int — 感測器數量
    n_samples : int — 每個狀態的取樣數
    n_modes : int — 主要振動模態數
    damage_ratio : float — 損傷時某模態變化比例

    Returns
    -------
    X_normal : ndarray, shape (n_samples, n_sensors)
    X_damaged : ndarray, shape (n_samples, n_sensors)
    mode_shapes : ndarray, shape (n_sensors, n_modes)
    """
    rng = np.random.default_rng(seed)

    # 建立模態形狀
    x_pos = np.linspace(0, np.pi, n_sensors)
    mode_shapes = np.zeros((n_sensors, n_modes))
    for i in range(n_modes):
        mode_shapes[:, i] = np.sin((i + 1) * x_pos)

    # 正常狀態
    amplitudes_normal = rng.normal(0, 1, (n_samples, n_modes))
    amplitudes_normal[:, 0] *= 3.0  # 第一模態最強
    amplitudes_normal[:, 1] *= 1.5
    X_normal = amplitudes_normal @ mode_shapes.T + rng.normal(0, 0.1, (n_samples, n_sensors))

    # 損傷狀態（第二模態變化）
    amplitudes_damaged = rng.normal(0, 1, (n_samples, n_modes))
    amplitudes_damaged[:, 0] *= 3.0
    amplitudes_damaged[:, 1] *= 1.5 * (1 + damage_ratio)
    # 損傷改變部分模態形狀
    damaged_modes = mode_shapes.copy()
    damaged_modes[n_sensors//3:n_sensors//2, 1] *= (1 - damage_ratio)
    X_damaged = amplitudes_damaged @ damaged_modes.T + rng.normal(0, 0.1, (n_samples, n_sensors))

    return X_normal, X_damaged, mode_shapes


def generate_drag_data(n_points=40, Cd_A=0.5, rho=1.225, noise_std=0.5,
                       v_range=(1, 30), seed=42):
    """模擬空氣阻力數據 Fd = 0.5 * rho * Cd * A * v^2

    Returns
    -------
    v : ndarray — 速度 (m/s)
    Fd : ndarray — 阻力 (N)
    true_params : dict
    """
    rng = np.random.default_rng(seed)
    v = np.linspace(v_range[0], v_range[1], n_points)
    Fd = 0.5 * rho * Cd_A * v ** 2 + rng.normal(0, noise_std, n_points)
    return v, Fd, {'Cd_A': Cd_A, 'rho': rho}


def generate_gps_measurements(true_pos, n_satellites=6, noise_std=5.0, seed=42):
    """模擬 GPS 衛星距離量測（2D）

    Parameters
    ----------
    true_pos : array-like, shape (2,) — 真實位置
    n_satellites : int — 衛星數
    noise_std : float — 距離量測噪聲 (m)

    Returns
    -------
    sat_positions : ndarray, shape (n_satellites, 2)
    measured_distances : ndarray, shape (n_satellites,)
    true_pos : ndarray
    """
    rng = np.random.default_rng(seed)
    true_pos = np.asarray(true_pos, dtype=float)

    # 衛星位置 — 分佈在目標周圍
    angles = np.linspace(0, 2 * np.pi, n_satellites, endpoint=False)
    radius = 20000  # km
    sat_positions = np.column_stack([
        true_pos[0] + radius * np.cos(angles),
        true_pos[1] + radius * np.sin(angles),
    ])

    true_distances = np.sqrt(np.sum((sat_positions - true_pos) ** 2, axis=1))
    measured_distances = true_distances + rng.normal(0, noise_std, n_satellites)

    return sat_positions, measured_distances, true_pos


def generate_vibration_signal(n_samples=1024, dt=0.001, frequencies=None,
                              amplitudes=None, noise_std=0.5, seed=42):
    """模擬帶噪聲的振動訊號

    Returns
    -------
    t : ndarray — 時間
    signal_clean : ndarray — 乾淨訊號
    signal_noisy : ndarray — 含噪訊號
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    if frequencies is None:
        frequencies = [10.0, 25.0, 50.0]
    if amplitudes is None:
        amplitudes = [3.0, 1.5, 0.8]

    signal_clean = np.zeros(n_samples)
    for f, a in zip(frequencies, amplitudes):
        signal_clean += a * np.sin(2 * np.pi * f * t)

    signal_noisy = signal_clean + rng.normal(0, noise_std, n_samples)
    return t, signal_clean, signal_noisy

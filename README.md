# 工程線性代數互動教程

> 從高中數學出發，用 Python 帶你走進線性代數的工程世界

## 這個教程是什麼？

你可能在課本上看過矩陣、向量這些詞，但總覺得「學這個有什麼用？」

這個教程會告訴你答案——線性代數是幾乎所有工程領域的共同語言：

- 機器手臂怎麼知道關節要轉多少度？ → **矩陣乘法**
- 橋梁能承受多少重量？ → **線性方程組**
- 大樓地震時怎麼搖晃？ → **特徵值**
- 感測器資料怎麼去除雜訊？ → **PCA 降維**

每個概念都搭配**物理實例**和**可執行的 Python 程式碼**，讓你看得見、摸得著。

## 適合誰？

- 高中生或大一新生，具備基本數學能力（三角函數、聯立方程式）
- 對物理或工程有興趣，想知道「數學到底能幹嘛」
- 想學 Python 科學計算的入門者

## 如何開始？

### 第一步：安裝環境

你需要 Python 3.9 以上版本。打開終端機（Terminal），輸入：

```bash
# 下載專案
git clone https://github.com/ChunPingWang/engineering-linalg-tutorial.git
cd engineering-linalg-tutorial

# 安裝套件
pip install -r requirements.txt
```

### 第二步：啟動 Jupyter

```bash
jupyter lab
```

瀏覽器會自動打開，進入 `notebooks/` 資料夾，從 `M1` 開始閱讀。

### 第三步：按照順序學習

每個 Notebook 都可以直接執行（按 `Shift + Enter`），邊看邊跑。

## 模組總覽

| 模組 | 主題 | 你會學到 | 物理例子 |
|------|------|----------|----------|
| **M1** | 向量基礎 | 向量加法、內積、外積、範數 | 拋體運動、力矩、太陽能板角度 |
| **M2** | 矩陣運算 | 矩陣乘法、旋轉、行列式、逆矩陣 | 機器手臂、座標轉換、感測器校正 |
| **M3** | 線性方程組 | 高斯消去法、最小平方法 | 桁架靜力分析、GPS 定位、熱電偶校正 |
| **M4** | 向量空間 | 線性獨立、秩、零空間、投影 | 機構自由度、剛體運動、加速度計去重力 |
| **M5** | 矩陣分解 | LU、QR、Cholesky、SVD | 有限元素法、影像壓縮、可操作性橢球 |
| **M6** | 特徵值 | 特徵值分解、穩定性分析 | 振動系統、倒擺控制、慣性張量 |
| **M7** | 最小平方 | 線性迴歸、正規化 | 彈簧常數擬合、空氣阻力、逆熱傳 |
| **M8** | PCA 降維 | 共變異數矩陣、主成分分析 | IMU 資料、橋梁健康監測、訊號降噪 |
| **M9** | 綜合實戰 | FEM、動力學、CFD、控制 | 桿件分析、頻率響應、溫度場、倒擺控制 |

## 專案結構

```
engineering-linalg-tutorial/
├── notebooks/                  # 九個教學 Notebook（主要閱讀這裡）
│   ├── M1_vectors_kinematics.ipynb
│   ├── M2_matrices_transforms.ipynb
│   ├── M3_linear_systems_statics.ipynb
│   ├── M4_vector_spaces_dof.ipynb
│   ├── M5_decompositions_engineering.ipynb
│   ├── M6_eigenvalues_dynamics.ipynb
│   ├── M7_least_squares_fitting.ipynb
│   ├── M8_pca_sensor.ipynb
│   └── M9_applications_capstone.ipynb
├── src/                        # 核心程式碼
│   ├── linalg_utils.py         # 手動實作的線代算法（教學用）
│   ├── physics_models.py       # 物理系統模型
│   ├── visualizer.py           # 視覺化工具
│   └── datasets.py             # 模擬感測器資料
├── tests/                      # 測試案例（89 個測試全部通過）
│   ├── test_vectors.py
│   ├── test_matrices.py
│   ├── test_decompositions.py
│   └── test_physics_models.py
├── requirements.txt
└── README.md
```

## 建議學習路線

### 第一週：打好基礎

先完成 M1 和 M2，這是所有後續內容的基礎。重點理解：
- 向量就是「有方向的量」，可以表示力、速度、位置
- 矩陣就是「線性轉換」，可以旋轉、縮放、映射

### 第二週：方程式與空間

完成 M3 和 M4。這時你會發現：
- 工程問題常常歸結為「解聯立方程式」
- 「秩」告訴你系統有多少獨立的約束

### 第三週：分解的力量

M5 是核心中的核心，特別是 SVD。你會看到：
- 同一個矩陣可以被「拆解」成不同形式，各有用途
- SVD 可以用來壓縮圖片、分析機器手臂的靈活度

### 第四週：進階應用

M6-M9 把所有工具串起來解決真實工程問題：
- 地震如何影響建築？（特徵值 → 固有頻率）
- 如何從嘈雜的資料中找到有用的訊號？（PCA）
- 如何讓不穩定的倒擺站住？（控制理論）

## 每個概念的驗證方式

本教程的每個數學計算都有驗證：

| 類型 | 驗證方法 |
|------|----------|
| 手動實作的算法 | 與 NumPy/SciPy 結果比對（`np.allclose`） |
| 物理模型 | 與解析解比對（如拋體運動、熱傳導線性分佈） |
| 矩陣性質 | 正交性 `R^TR=I`、重組誤差 `< 1e-12` |
| 物理合理性 | 力平衡、能量守恆、穩定性條件 |

執行測試：

```bash
python -m pytest tests/ -v
```

所有 89 個測試案例均通過。

## 使用的套件

| 套件 | 用途 |
|------|------|
| NumPy | 核心數值運算（向量、矩陣） |
| SciPy | 進階線代、微分方程、稀疏矩陣 |
| Matplotlib | 2D/3D 視覺化（圖表、動畫） |
| SymPy | 符號運算 |
| pandas | 資料處理 |
| scikit-learn | PCA 驗證比對 |
| JupyterLab | 互動式筆記本 |

## 常用公式速查

| 數學 | Python | 說明 |
|------|--------|------|
| a . b | `np.dot(a, b)` | 內積 |
| a x b | `np.cross(a, b)` | 外積 |
| \|\|v\|\| | `np.linalg.norm(v)` | 向量長度 |
| AB | `A @ B` | 矩陣乘法 |
| A^(-1) | `np.linalg.inv(A)` | 逆矩陣 |
| det(A) | `np.linalg.det(A)` | 行列式 |
| Ax = b | `np.linalg.solve(A, b)` | 解方程組 |
| A = USV^T | `np.linalg.svd(A)` | SVD 分解 |
| Av = λv | `np.linalg.eig(A)` | 特徵值 |

## 授權

本教程為開源教學專案，歡迎自由使用與分享。

import numpy as np
from numba import njit
import time

# ------------------------------------------------------------
# 1. 纯 Python 版本
# ------------------------------------------------------------
def sum_of_distances_python(points):
    total = 0.0
    for x, y in points:
        total += (x*x + y*y)**0.5
    return total

# ------------------------------------------------------------
# 2. Numba 加速版本
# ------------------------------------------------------------
@njit
def sum_of_distances_numba(points):
    total = 0.0
    n = points.shape[0]
    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        total += (x*x + y*y)**0.5
    return total

# ------------------------------------------------------------
# 3. NumPy 向量化版本（原地内存分配，使用 float64）
# ------------------------------------------------------------
def sum_of_distances_numpy_inplace_f64(points):
    """
    完全原地操作，points 必须是 float64 类型。
    先平方，再求和，再开方，最后求和。
    所有步骤均使用 float64，保证与 Python/Numba 精度一致。
    """
    # 原地平方：points = points^2
    np.square(points, out=points)
    # 沿 axis=1 求和，结果保存在一维数组 d2 中（float64）
    d2 = np.sum(points, axis=1, dtype=np.float64)
    # 原地开平方：d2 = sqrt(d2)
    np.sqrt(d2, out=d2)
    # 返回总和（已经是 float64）
    return d2.sum()

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    n_points = 10_000_000
    dtype = np.float64   # 使用双精度
    print(f"生成 {n_points:,} 个随机点（dtype={dtype}）...")
    points_original = np.random.randn(n_points, 2).astype(dtype)

    # --- 纯 Python 版本 ---
    points_py = points_original.copy()
    start = time.perf_counter()
    result_py = sum_of_distances_python(points_py)
    time_py = time.perf_counter() - start
    print(f"纯 Python 结果: {result_py:.15f}，耗时: {time_py:.3f} 秒")

    # --- Numba 版本 ---
    points_nb = points_original.copy()
    _ = sum_of_distances_numba(points_nb)   # 预热
    start = time.perf_counter()
    result_nb = sum_of_distances_numba(points_nb)
    time_nb = time.perf_counter() - start
    print(f"Numba 结果:     {result_nb:.15f}，耗时: {time_nb:.3f} 秒")

    # --- NumPy 原地版本（float64）---
    points_np = points_original.copy()
    start = time.perf_counter()
    result_np = sum_of_distances_numpy_inplace_f64(points_np)
    time_np = time.perf_counter() - start
    print(f"NumPy 原地结果: {result_np:.15f}，耗时: {time_np:.3f} 秒")

    print(f"\n加速比 (相对纯 Python):")
    print(f"Numba:  {time_py / time_nb:.1f}x")
    print(f"NumPy:  {time_py / time_np:.1f}x")

    # 校验：双精度下误差应小于 1e-12
    assert abs(result_py - result_nb) < 1e-12, "Numba 结果不一致"
    assert abs(result_py - result_np) < 1e-12, "NumPy 结果不一致"
    print("\n数值校验通过，三种方法结果完全一致（双精度）。")
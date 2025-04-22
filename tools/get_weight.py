import numpy as np

# 定义向量 A, B, C
A = np.array([3630.895614, 3281.933196, 7.365149, 0.003619, 0.004469, 0.790201, -0.612821])
B = np.array([3606.992044, 3288.080863, 17.265990,0.004836, -0.003570, 0.787337, -0.616493])
C = np.array([3610.8319, 3300.7233, 13.6472, 0.0014, -0.0055, 0.7873, -0.6166])

# 构建系数矩阵 M，其中每一列是一个向量
# M 的形状是 7x2 (7行，2列)
M = np.vstack([A, B]).T  # .T 表示转置，使得 A 和 B 成为列向量

# 使用 numpy.linalg.lstsq 求解最小二乘问题
# 它求解的是 M * [x, y]^T = C 的最小二乘解
# result[0] 包含了求解出的 [x, y]
result = np.linalg.lstsq(M, C, rcond=None)
x_optimal, y_optimal = result[0]

print(f"最优解 x ≈ {x_optimal}")
print(f"最优解 y ≈ {y_optimal}")

# 你还可以查看残差（误差平方和）等信息
residuals = result[1]
print(f"残差平方和 ≈ {residuals[0] if residuals.size > 0 else '无（精确解）'}") # 如果残差数组不为空，则打印第一个元素

# 验证一下结果（可选）
C_calculated = A * x_optimal + B * y_optimal
print("\n计算得到的 C 值:")
print(C_calculated)
print("\n原始的 C 值:")
print(C)
print("\n差值 (误差):")
print(C_calculated - C)
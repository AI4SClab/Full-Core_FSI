import numpy as np
from TreeClass import MeshPoint
import math
from scipy.optimize import minimize


def solve_weights_least_squares(rbf_matrix, parameters):
    """使用最小二乘法求解 W"""
    W, residuals, rank, s = np.linalg.lstsq(rbf_matrix, parameters, rcond=None)
    return W
# def solve_weights_least_squares(rbf_matrix, parameters):
#     """
#     使用最小二乘法求解 W，并处理病态矩阵或奇异矩阵的情况
#     """
#     try:
#         # 检查矩阵是否存在异常值
#         if np.isnan(rbf_matrix).any() or np.isinf(rbf_matrix).any():
#             raise ValueError("rbf_matrix contains NaN or Inf values")
        
#         # 检查矩阵的条件数
#         cond_number = np.linalg.cond(rbf_matrix)
#         print("Condition number of rbf_matrix:", cond_number)
#         if cond_number > 1e10:
#             print("Warning: rbf_matrix is ill-conditioned, adding regularization")
#             epsilon = 1e-10
#             rbf_matrix += epsilon * np.eye(rbf_matrix.shape[1])
        
#         # 尝试用最小二乘法求解
#         W, residuals, rank, s = np.linalg.lstsq(rbf_matrix, parameters, rcond=1e-10)
#         return W
    
#     except np.linalg.LinAlgError as e:
#         print("LinAlgError:", e, "Using pseudo-inverse instead")
#         # 使用伪逆作为后备方案
#         R_pinv = np.linalg.pinv(rbf_matrix)
#         return R_pinv @ parameters

def solve_weights_linear_algebra(rbf_matrix, parameters):
    """使用线性代数求解 W（仅在 rbf_matrix 可逆时有效）"""
    return np.linalg.inv(rbf_matrix) @ parameters


def solve_weights_qr(rbf_matrix, parameters):
    """使用 QR 分解求解 W"""
    Q, R = np.linalg.qr(rbf_matrix)
    return np.linalg.solve(R, Q.T @ parameters)
# def solve_weights_qr(rbf_matrix, parameters):
#     """
#     使用 QR 分解求解 W，并处理奇异矩阵的情况
#     """
#     # QR 分解
#     Q, R = np.linalg.qr(rbf_matrix)
    
#     # 检查 R 是否为奇异矩阵
#     if np.linalg.matrix_rank(R) < R.shape[0]:
#         print("Warning: R is singular, using pseudo-inverse")
#         R_pinv = np.linalg.pinv(R)  # 使用伪逆
#         return R_pinv @ (Q.T @ parameters)
    
#     try:
#         # 正常求解
#         return np.linalg.solve(R, Q.T @ parameters)
#     except np.linalg.LinAlgError as e:
#         # 如果 solve 失败，使用伪逆作为后备方案
#         print("LinAlgError:", e, "Using pseudo-inverse instead")
#         R_pinv = np.linalg.pinv(R)
#         return R_pinv @ (Q.T @ parameters)


def solve_weights_ridge(rbf_matrix, parameters, alpha=1e-5):
    """使用岭回归求解 W"""
    return (
        np.linalg.inv(rbf_matrix.T @ rbf_matrix + alpha * np.eye(rbf_matrix.shape[1]))
        @ rbf_matrix.T
        @ parameters
    )


def gradient_descent(rbf_matrix, parameters, learning_rate=0.01, num_iterations=1000):
    """使用梯度下降法求解 W"""
    m, n = rbf_matrix.shape
    W = np.zeros((n, 1))  # 初始化权重
    for _ in range(num_iterations):
        gradient = (1 / m) * rbf_matrix.T @ (rbf_matrix @ W - parameters)
        W -= learning_rate * gradient
    return W


def solve_weights_optimization(rbf_matrix, parameters):
    """使用优化库求解 W"""

    def objective_function(W):
        return np.sum((rbf_matrix @ W - parameters.flatten()) ** 2)

    initial_W = np.zeros((rbf_matrix.shape[1], 1))
    result = minimize(objective_function, initial_W)
    return result.x.reshape(-1, 1)






def rbf_function(fj: MeshPoint, si: MeshPoint, R: float):
    """根据给定的 fj 和 si 计算 RBF 值, R为支撑半径"""
    # 计算 η
    eta = math.sqrt((fj.x - si.x) ** 2 + (fj.y - si.y) ** 2 + (fj.z - si.z) ** 2) / R
    # 根据 η 计算 RBF
    if eta <= 1:
        return (1 - eta) ** 4 * (4 * eta + 1)
    else:
        return 0


def compute_rbf_matrix(solid_points: [MeshPoint], fluid_points: [MeshPoint], R):
    """构建 RBF 矩阵"""
    num1 = len(solid_points)  # 固体点数量
    num2 = len(fluid_points)  # 流体点数量

    # 初始化 RBF 矩阵
    rbf_matrix = np.zeros((num1, num2))

    for i, si in enumerate(solid_points):  # 第一个元素是当前元素的索引（从 0 开始）。
        for j, fj in enumerate(fluid_points):
            # 计算 RBF 值并存储在矩阵中
            rbf_matrix[i, j] = rbf_function(si, fj, R)

    return rbf_matrix


def compute_fluidrbf_matrix(fluid_points: [MeshPoint], R):
    """构建 RBF 矩阵"""
    num2 = len(fluid_points)  # 流体点数量

    # 初始化 RBF 矩阵
    rbf_matrix_f = np.zeros((num2, num2))
    pressure_matrix_f = np.zeros((num2, 1))

    for i, fi in enumerate(fluid_points):  # 第一个元素是当前元素的索引（从 0 开始）。
        for j, fj in enumerate(fluid_points):
            # 计算 RBF 值并存储在矩阵中
            rbf_matrix_f[i, j] = rbf_function(fi, fj, R)

            if i == 0:
                pressure_matrix_f[j, 0] = fj.pressure

    return rbf_matrix_f, pressure_matrix_f


def compute_solidrbf_matrix(solid_points: [MeshPoint], R):
    """构建 RBF 矩阵"""
    num1 = len(solid_points)  # 流体点数量

    # 初始化 RBF 矩阵
    rbf_matrix_s = np.zeros((num1, num1))
    velocity0_matrix_s = np.zeros((num1, 1))
    velocity1_matrix_s = np.zeros((num1, 1))
    velocity2_matrix_s = np.zeros((num1, 1))

    for i, fi in enumerate(solid_points):  # 第一个元素是当前元素的索引（从 0 开始）。
        for j, fj in enumerate(solid_points):
            # 计算 RBF 值并存储在矩阵中
            rbf_matrix_s[i, j] = rbf_function(fi, fj, R)

            if i == 0:
                velocity0_matrix_s[j, 0] = fj.vx
                velocity1_matrix_s[j, 0] = fj.vy
                velocity2_matrix_s[j, 0] = fj.vz

    return rbf_matrix_s, velocity0_matrix_s, velocity1_matrix_s, velocity2_matrix_s


def compute_pressure(solid_points: [MeshPoint], fluid_points: [MeshPoint], R):
    # 根据流体节点计算RBF矩阵
    rbf_matrix_f, pressure_matrix_f = compute_fluidrbf_matrix(fluid_points, R)
    # 使用最小二乘法求解 W
    Wf = solve_weights_least_squares(rbf_matrix_f, pressure_matrix_f)
    # Wf = solve_weights_qr(rbf_matrix_f, pressure_matrix_f)
    # 为固体节点计算压力矩阵
    rbf_matrix_s = compute_rbf_matrix(solid_points, fluid_points, R)

    pressure_matrix_s = rbf_matrix_s @ Wf
    return pressure_matrix_s


def compute_velocity(fluid_points: [MeshPoint], solid_points: [MeshPoint], R):
    # 根据固体节点计算RBF矩阵
    rbf_matrix_s, velocity0_matrix_s, velocity1_matrix_s, velocity2_matrix_s = (
        compute_solidrbf_matrix(solid_points, R)
    )
    # 使用最小二乘法求解 W
    Wsx = solve_weights_least_squares(rbf_matrix_s, velocity0_matrix_s)
    Wsy = solve_weights_least_squares(rbf_matrix_s, velocity1_matrix_s)
    Wsz = solve_weights_least_squares(rbf_matrix_s, velocity2_matrix_s)
    # Wsx = solve_weights_qr(rbf_matrix_s, velocity0_matrix_s)
    # Wsy = solve_weights_qr(rbf_matrix_s, velocity1_matrix_s)
    # Wsz = solve_weights_qr(rbf_matrix_s, velocity2_matrix_s)
    

    # 为流体节点计算速度分量
    rbf_matrix_f = compute_rbf_matrix(fluid_points, solid_points, R)
    velocity0_matrix_f = rbf_matrix_f @ Wsx
    velocity1_matrix_f = rbf_matrix_f @ Wsy
    velocity2_matrix_f = rbf_matrix_f @ Wsz

    return velocity0_matrix_f, velocity1_matrix_f, velocity2_matrix_f

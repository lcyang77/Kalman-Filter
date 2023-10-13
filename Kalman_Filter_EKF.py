import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 时间步长
dt = 0.1

# 总的模拟时间
T = 10

# 时间步数
steps = int(T / dt)

# 初始状态
x_true = np.array([0, 0, np.pi/2]).reshape(3, 1)

# 状态估计
x_est = x_true.copy()

# 初始协方差矩阵
P = np.eye(3)

# 过程噪声协方差
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)])**2

# 测量噪声协方差
R = np.diag([0.5, 0.5])**2

# 控制输入 [v, omega]
u = np.array([1, 0.1])

# EKF 预测步骤
def ekf_predict(x_est, P, u, Q, dt):
    # 预测状态
    x_pred = x_est.copy()
    x_pred[0] += dt * u[0] * np.cos(x_est[2])
    x_pred[1] += dt * u[0] * np.sin(x_est[2])
    x_pred[2] += dt * u[1]

    # 计算运动模型的雅可比矩阵
    F = np.array([[1, 0, -dt * u[0] * np.sin(x_est[2, 0])],
                  [0, 1, dt * u[0] * np.cos(x_est[2, 0])],
                  [0, 0, 1]])

    # 预测协方差
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred

# EKF 更新步骤
def ekf_update(x_pred, P_pred, z, R):
    # 计算测量模型的雅可比矩阵
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])

    # 计算卡尔曼增益
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)

    # 更新状态
    x_est = x_pred + K @ (z.reshape(2, 1) - H @ x_pred)

    # 更新协方差
    P_est = (np.eye(3) - K @ H) @ P_pred

    return x_est, P_est

# 用于存储真实轨迹和测量的列表
trajectory_true = []
measurements = []
estimates = []

for _ in range(steps):
    # 模拟真实状态
    x_true[0] += dt * u[0] * np.cos(x_true[2])
    x_true[1] += dt * u[0] * np.sin(x_true[2])
    x_true[2] += dt * u[1]
    trajectory_true.append(x_true.copy())

    # 生成带有噪声的测量
    z = np.array([x_true[0], x_true[1]]) + np.sqrt(R) @ np.random.randn(2, 1)
    measurements.append(z.copy())

    # EKF 预测
    x_pred, P_pred = ekf_predict(x_est, P, u, Q, dt)

    # EKF 更新
    x_est, P = ekf_update(x_pred, P_pred, z, R)
    estimates.append(x_est.copy())

# 将列表转换为 numpy 数组，以便绘图
trajectory_true = np.array(trajectory_true).squeeze()
measurements = np.array(measurements).squeeze()
estimates = np.array(estimates).squeeze()

# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(trajectory_true[:, 0], trajectory_true[:, 1], label='真实轨迹')
plt.scatter(measurements[:, 0], measurements[:, 1], label='测量', c='red', s=5)
plt.plot(estimates[:, 0], estimates[:, 1], label='EKF 估计', c='green')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('EKF 状态估计')
plt.grid(True)
plt.show()

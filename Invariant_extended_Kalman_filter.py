import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义非线性动力学模型
def dynamics_model(state, omega, dt):
    return np.array([
        state[0] + state[2] * np.cos(state[3]) * dt,
        state[1] + state[2] * np.sin(state[3]) * dt,
        state[2],
        state[3] + omega * dt
    ])


# 定义非线性测量模型
def measurement_model(state, sensor_pos):
    return np.sqrt((state[0] - sensor_pos[0]) ** 2 + (state[1] - sensor_pos[1]) ** 2)


# 定义 IEKF 更新函数
def iekf_update(state, P, z_meas, sensor_pos, Q, R, dt):
    for _ in range(2):  # 进行两次迭代
        # 预测
        state_pred = dynamics_model(state, 0, dt)  # 零角速度预测

        # 计算雅可比矩阵
        x_pred = state_pred[0]
        y_pred = state_pred[1]
        Hx = (x_pred - sensor_pos[0]) / np.sqrt((x_pred - sensor_pos[0]) ** 2 + (y_pred - sensor_pos[1]) ** 2)
        Hy = (y_pred - sensor_pos[1]) / np.sqrt((x_pred - sensor_pos[0]) ** 2 + (y_pred - sensor_pos[1]) ** 2)
        H = np.array([Hx, Hy, 0, 0])

        # 计算卡尔曼增益
        S = H @ P @ H.T + R
        K = P @ H.T / S

        # 状态更新
        z_pred = measurement_model(state_pred, sensor_pos)
        z_diff = z_meas - z_pred
        state = state_pred + K * z_diff

        # 协方差更新
        P = P - np.outer(K * S, K.T)

    return state, P


# 模拟参数
dt = 0.1  # 时间步长
steps = 200  # 时间步数

# 初始化状态
true_state = np.array([0.0, 0.0, 1.0, np.pi / 4])
estimated_state = np.array([0.0, 0.0, 1.0, np.pi / 4])
P = np.eye(4)  # 协方差矩阵初始化
Q = np.eye(4) * 0.01  # 过程噪声协方差
R = 0.1  # 测量噪声协方差

# 传感器位置
sensor_pos = np.array([5.0, 5.0])

# 存储轨迹和测量值
true_trajectory = np.zeros((2, steps))
estimated_trajectory = np.zeros((2, steps))
measurements = np.zeros(steps)

# 模拟
for step in range(steps):
    omega = np.random.normal(0, 0.1)  # 生成角速度噪声
    w = np.random.normal(0, np.sqrt(R))  # 生成测量噪声

    true_state = dynamics_model(true_state, omega, dt)  # 真实状态更新

    z_true = measurement_model(true_state, sensor_pos)  # 得到真实测量值
    z_meas = z_true + w  # 添加噪声

    estimated_state, P = iekf_update(estimated_state, P, z_meas, sensor_pos, Q, R, dt)  # IEKF 更新

    true_trajectory[:, step] = true_state[:2]  # 存储真实轨迹
    estimated_trajectory[:, step] = estimated_state[:2]  # 存储估计轨迹
    measurements[step] = z_meas  # 存储测量值

# 绘图
plt.plot(true_trajectory[0, :], true_trajectory[1, :], label='真实轨迹')
plt.plot(estimated_trajectory[0, :], estimated_trajectory[1, :], label='估计轨迹')
plt.plot(sensor_pos[0], sensor_pos[1], 'ro', label='传感器位置')
plt.xlabel('X 位置')
plt.ylabel('Y 位置')
plt.title('IEKF 非线性系统')
plt.grid(True)
plt.legend()
plt.show()

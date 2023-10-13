import numpy as np
import matplotlib.pyplot as plt


# 定义卡尔曼滤波器
def kalman_filter(z, F, H, Q, R, x0, P0):
    x = np.zeros((x0.shape[0], z.shape[1]))
    P = np.zeros((P0.shape[0], P0.shape[1], z.shape[1]))
    x[:, 0] = x0.ravel()
    P[:, :, 0] = P0

    for i in range(1, z.shape[1]):
        # 预测步骤
        x_pred = np.dot(F, x[:, i - 1])
        P_pred = np.dot(np.dot(F, P[:, :, i - 1]), F.T) + Q

        # 更新步骤
        y = z[:, i] - np.dot(H, x_pred)
        S = np.dot(np.dot(H, P_pred), H.T) + R
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
        x[:, i] = (x_pred + np.dot(K, y)).ravel()
        P[:, :, i] = P_pred - np.dot(np.dot(K, S), K.T)

    return x, P


# 模拟参数
dt = 0.1
N = 200
sigma_a = 0.1
sigma_z = 5

# 状态转移矩阵
F = np.array([[1, dt], [0, 1]])

# 过程噪声协方差矩阵
G = np.array([[0.5 * dt ** 2], [dt]])
Q = np.dot(G, G.T) * sigma_a ** 2

# 测量矩阵
H = np.array([[1, 0]])

# 测量噪声协方差矩阵
R = np.array([[sigma_z ** 2]])

# 初始状态
x0 = np.array([[0], [0]])

# 生成卡车的真实位置和速度
np.random.seed(0)
a = np.random.randn(N) * sigma_a
x_true = np.zeros((2, N))
x_true[:, 0] = x0.ravel()
for i in range(1, N):
    x_true[:, i] = np.dot(F, x_true[:, i - 1]) + np.dot(G, a[i - 1]).ravel()

# 生成卡车位置的噪声测量值
z = np.dot(H, x_true) + np.random.randn(1, N) * sigma_z

# 初始协方差矩阵
P0 = np.array([[0, 0], [0, 0]])

# 执行卡尔曼滤波
x_est, P_est = kalman_filter(z, F, H, Q, R, x0, P0)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x_true[0], label='真实位置')
plt.plot(z[0], 'o', markersize=4, label='测量位置')
plt.plot(x_est[0], label='卡尔曼滤波估计位置')
plt.xlabel('时间步')
plt.ylabel('位置')
plt.title('卡车位置的真实值、测量值和卡尔曼滤波估计值')
plt.legend()
plt.grid(True)
plt.show()

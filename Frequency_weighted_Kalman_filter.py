import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 生成模拟数据
np.random.seed(0)  # 设置随机种子，确保结果可复现

# 定义时间步长和总的时间步数
dt = 1.0
steps = 100

# 定义初始状态 [位置, 速度]
x = np.array([0, 1], dtype=float)

# 定义状态转移矩阵
F = np.array([[1, dt], [0, 1]])

# 定义控制输入矩阵
B = np.zeros_like(F)

# 定义控制向量
u = np.zeros_like(x)

# 定义过程噪声协方差
Q = np.array([[0.01, 0], [0, 0.01]])

# 定义观测模型矩阵
H = np.array([[1, 0]])

# 定义观测噪声协方差
R = np.array([[0.1]])

# 生成真实状态和带噪声的观测数据
true_states = np.zeros((steps, 2))
noisy_measurements = np.zeros(steps)

for k in range(steps):
    true_states[k] = x
    x = F.dot(x) + B.dot(u) + np.random.multivariate_normal([0, 0], Q)
    z = H.dot(x) + np.random.normal(0, np.sqrt(R))
    noisy_measurements[k] = z


# 2. 定义卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F  # 状态转移模型
        self.B = B  # 控制输入模型
        self.H = H  # 观测模型
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.x = x0  # 初始状态估计
        self.P = P0  # 初始估计误差协方差

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.identity(self.F.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x


# 3. 应用卡尔曼滤波器
# 定义初始估计误差协方差
P0 = np.identity(2)

# 应用标准卡尔曼滤波器
kf_standard = KalmanFilter(F, B, H, Q, R, x0=true_states[0], P0=P0)
standard_estimates = np.zeros((steps, 2))

for k in range(steps):
    kf_standard.predict(u)
    standard_estimates[k] = kf_standard.update(noisy_measurements[k])

# 应用频率加权卡尔曼滤波器，通过增加观测噪声协方差来实现
R_weighted = R * 10
kf_weighted = KalmanFilter(F, B, H, Q, R_weighted, x0=true_states[0], P0=P0)
weighted_estimates = np.zeros((steps, 2))

for k in range(steps):
    kf_weighted.predict(u)
    weighted_estimates[k] = kf_weighted.update(noisy_measurements[k])


# 4. 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], label='真实位置')
plt.plot(noisy_measurements, label='带噪声的观测', linestyle='--')
plt.plot(standard_estimates[:, 0], label='标准卡尔曼滤波器估计')
plt.plot(weighted_estimates[:, 0], label='频率加权卡尔曼滤波器估计', linestyle='-.')
plt.xlabel('时间步')
plt.ylabel('位置')
plt.title('卡尔曼滤波器估计')
plt.legend()
plt.grid(True)
plt.show()

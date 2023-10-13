import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义非线性函数
def f(x, dt):
    return x + np.sin(x) * dt

# 定义 UKF 的初始状态和噪声
x0 = np.array([0.])
Q = np.diag([0.01**2])   # 系统噪声的协方差
R = 0.1**2               # 测量噪声的协方差

# 定义 Unscented Kalman Filter
points = MerweScaledSigmaPoints(1, alpha=1e-3, beta=2, kappa=0)
ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1,
                            fx=f, hx=lambda x: x,
                            points=points)
ukf.x = x0
ukf.Q = Q
ukf.R = R

# 生成模拟数据
dt = 1
steps = 100
x = np.zeros((steps, 1))
x[0] = x0
for t in range(1, steps):
    x[t] = f(x[t-1], dt) + np.sqrt(Q) * np.random.randn(1, 1)

# 添加测量噪声
z = x + np.sqrt(R) * np.random.randn(steps, 1)

# 使用 UKF 进行滤波
ukf_xs = np.zeros((steps, 1))
for t in range(steps):
    ukf.predict()
    ukf.update(z[t])
    ukf_xs[t] = ukf.x

# 绘制结果
plt.figure()
plt.plot(x, label='真实值')
plt.plot(z, 'o', label='测量值')
plt.plot(ukf_xs, label='UKF估计值')
plt.legend()
plt.show()


# Step 1: 生成信号

np.random.seed(0)    # 设置随机种子，保证结果可重复
N = 500               # 样本数量
t = np.linspace(0, 1, N)  # 时间向量
freq = 5             # 正弦波频率
sine_wave = np.sin(2 * np.pi * freq * t)  # 生成正弦波
noise = 0.5 * np.random.randn(N)  # 生成随机噪声
signal = sine_wave + noise  # 得到含噪声的信号

# Step 2: 实现RLS滤波器

class RLSFilter:
    def __init__(self, filter_length, lambda_, delta_):
        self.filter_length = filter_length  # 滤波器长度
        self.lambda_ = lambda_  # λ参数
        # 初始化权重矩阵为零
        self.weights = np.zeros(filter_length)
        # 初始化误差协方差矩阵为delta倍的单位矩阵
        self.P = delta_ * np.eye(filter_length)

    def update(self, d, x):
        x = np.array(x)
        # 使用权重计算估计输出
        y = np.dot(self.weights, x)
        # 更新误差协方差矩阵
        k = self.P @ x / (self.lambda_ + x.T @ self.P @ x)
        self.P = (self.P - np.outer(k, x.T @ self.P)) / self.lambda_
        # 使用预测误差更新权重
        self.weights += k * (d - y)
        return y

# 应用RLS滤波器到信号上

filter_length = 4
lambda_ = 0.99
delta_ = 1.0

rls_filter = RLSFilter(filter_length, lambda_, delta_)
estimated_signal = []

for i in range(filter_length, N):
    x = signal[i-filter_length:i]  # 获取当前输入
    d = signal[i]  # 获取当前期望输出
    y = rls_filter.update(d, x)  # 更新滤波器并得到估计输出
    estimated_signal.append(y)

# Step 3: 可视化结果

plt.figure(figsize=(14, 6))
plt.plot(t, signal, label='原始信号', alpha=0.7)
plt.plot(t[filter_length:], estimated_signal, 'r', label='RLS估计')
plt.title('原始信号 vs. RLS估计')
plt.legend()
plt.grid(True)
plt.show()

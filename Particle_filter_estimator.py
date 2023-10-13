import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化
np.random.seed(0)  # 设置随机种子
T = 50  # 时间步数
numParticles = 1000  # 粒子数量

real_x = np.zeros(T)  # 真实状态
z = np.zeros(T)  # 观测
estimates = np.zeros(T)  # 估计

particles = np.random.randn(numParticles)  # 初始化粒子
weights = np.ones(numParticles) / numParticles  # 初始化权重

# 生成真实状态和观测
for t in range(1, T):
    real_x[t] = real_x[t - 1] + np.random.randn()  # 真实状态是一个随机漫步
    z[t] = real_x[t] + np.random.randn()  # 观测包含噪声

# 粒子滤波
for t in range(T):
    # 预测步骤
    particles += np.random.randn(numParticles)  # 添加过程噪声

    # 更新步骤
    innovation = z[t] - particles  # 计算创新（观测减去预测）
    likelihood = np.exp(-0.5 * innovation ** 2)  # 计算似然
    weights *= likelihood  # 更新权重
    weights /= weights.sum()  # 归一化权重

    # 状态估计
    estimates[t] = np.sum(particles * weights)  # 加权平均

    # 重采样
    indices = np.random.choice(range(numParticles), numParticles, p=weights)  # 基于权重进行重采样
    particles = particles[indices]  # 更新粒子集
    weights = np.ones(numParticles) / numParticles  # 重置权重

# 绘图
plt.figure()
plt.plot(range(T), real_x, 'b-', label='真实状态')  # 真实状态
plt.plot(range(T), z, 'k.', label='观测')  # 观测
plt.plot(range(T), estimates, 'r--', label='估计')  # 粒子滤波器估计
plt.legend()
plt.xlabel('时间步')
plt.ylabel('状态')
plt.title('使用粒子滤波器进行状态估计')
plt.grid(True)
plt.show()

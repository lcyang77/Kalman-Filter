import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 定义粒子滤波器类
class ParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles  # 粒子数量
        self.particles = np.random.normal(0, 1, num_particles)  # 初始化粒子
        self.weights = np.ones(num_particles) / num_particles  # 初始化权重

    def predict(self):
        # 预测步骤，粒子根据系统动力学和噪声进行演化
        noise = np.random.normal(0, 1, self.num_particles)  # 系统噪声
        self.particles += noise  # 粒子状态更新

    def update(self, z):
        # 更新步骤，根据观测数据更新粒子的权重
        noise = np.random.normal(0, 1, self.num_particles)  # 观测噪声
        likelihood = np.exp(-(z - (self.particles + noise)) ** 2 / 2)  # 计算似然
        self.weights = likelihood * self.weights  # 更新权重
        self.weights += 1.e-300  # 避免除以零
        self.weights /= sum(self.weights)  # 归一化权重

    def resample(self):
        # 重采样步骤，根据权重重新采样粒子
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)  # 采样索引
        self.particles = self.particles[indices]  # 更新粒子
        self.weights = np.ones(self.num_particles) / self.num_particles  # 重置权重

    def estimate(self):
        # 估计当前状态
        return np.mean(self.particles)  # 返回粒子均值作为状态估计

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 模拟一个简单的一维随机过程
np.random.seed(0)  # 设置随机种子
T = 50  # 总时间
real_x = np.zeros(T)  # 真实状态
z = np.zeros(T)  # 观测

# 生成真实状态和观测数据
for t in range(1, T):
    real_x[t] = real_x[t - 1] + np.random.normal()  # 真实状态
    z[t] = real_x[t] + np.random.normal()  # 带噪声的观测

# 使用粒子滤波器估计状态
pf = ParticleFilter(1000)  # 创建粒子滤波器实例，粒子数为1000
estimates = np.zeros(T)  # 估计值

# 进行滤波
for t in range(T):
    pf.predict()  # 预测
    pf.update(z[t])  # 更新
    estimates[t] = pf.estimate()  # 存储估计值
    pf.resample()  # 重采样

# 绘图展示结果
plt.figure(figsize=(10, 6))
plt.plot(real_x, label='真实状态')  # 真实状态
plt.plot(z, label='观测', linestyle='dotted')  # 观测
plt.plot(estimates, label='估计', linestyle='dashed')  # 粒子滤波器估计
plt.legend()  # 添加图例
plt.xlabel('时间')  # x轴标签
plt.ylabel('状态')  # y轴标签
plt.title('粒子滤波器状态估计')  # 图标题
plt.grid(True)  # 添加网格
plt.show()  # 展示图像

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy.random import normal, seed


# 定义洛伦茨系统的动力学方程
def lorenz(state, t, sigma, beta, rho):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


# 生成洛伦茨系统的真实轨迹
def generate_true_trajectory(initial_state, t, sigma, beta, rho):
    return odeint(lorenz, initial_state, t, args=(sigma, beta, rho))


# 添加噪声来模拟观测数据
def add_observation_noise(trajectory, noise_std):
    noisy_data = trajectory + normal(scale=noise_std, size=trajectory.shape)
    return noisy_data[:, :2]  # 假设我们只能观测到x和y


# 使用集合卡尔曼滤波器来估计状态
def ensemble_kalman_filter(observations, initial_ensemble, noise_std, t, sigma, beta, rho):
    num_ensemble_members = initial_ensemble.shape[0]
    state_estimate = np.zeros((len(t), 3))
    ensembles = np.zeros((len(t), num_ensemble_members, 3))
    ensembles[0] = initial_ensemble

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]

        # 预测步骤
        forecast_ensemble = np.array([
            odeint(lorenz, ensembles[i - 1, j], [0, dt], args=(sigma, beta, rho))[-1]
            for j in range(num_ensemble_members)
        ])

        ensembles[i] = forecast_ensemble

        # 更新步骤
        ensemble_mean = np.mean(forecast_ensemble, axis=0)
        observation_mean = np.mean(observations[i - 1:i + 1], axis=0)

        Y = forecast_ensemble[:, :2] + normal(scale=noise_std, size=(num_ensemble_members, 2))
        Y_mean = np.mean(Y, axis=0)
        Pxy = (forecast_ensemble - ensemble_mean).T @ (Y - Y_mean) / (num_ensemble_members - 1)
        Pyy = np.cov(Y, rowvar=False) + noise_std ** 2 * np.eye(2)
        K = np.linalg.solve(Pyy.T, Pxy.T).T

        ensembles[i] = forecast_ensemble + K @ (observation_mean - Y_mean)

        state_estimate[i] = np.mean(ensembles[i], axis=0)

    return state_estimate, ensembles


# 设置随机种子以确保结果的可重复性
seed(42)

# 定义初始状态和时间
initial_state = np.array([1, 1, 1])
t = np.linspace(0, 10, 1000)
sigma, beta, rho = 10, 8 / 3, 28

# 生成真实轨迹
true_trajectory = generate_true_trajectory(initial_state, t, sigma, beta, rho)

# 生成观测数据
observation_noise_std = 1.0
observations = add_observation_noise(true_trajectory, observation_noise_std)

# 定义初始集合
num_ensemble_members = 100
initial_ensemble = normal(scale=1, size=(num_ensemble_members, 3))

# 使用集合卡尔曼滤波器估计状态
state_estimate, ensembles = ensemble_kalman_filter(
    observations, initial_ensemble, observation_noise_std, t, sigma, beta, rho
)

# 绘制结果
plt.figure(figsize=(10, 8))

# 绘制 x 状态
plt.subplot(3, 1, 1)
plt.plot(t, true_trajectory[:, 0], 'k', label='True x')
plt.plot(t, observations[:, 0], '.r', label='Observed x')
plt.plot(t, state_estimate[:, 0], 'b', label='Estimated x')
plt.legend()

# 绘制 y 状态
plt.subplot(3, 1, 2)
plt.plot(t, true_trajectory[:, 1], 'k', label='True y')
plt.plot(t, observations[:, 1], '.r', label='Observed y')
plt.plot(t, state_estimate[:, 1], 'b', label='Estimated y')
plt.legend()

# 绘制 z 状态
plt.subplot(3, 1, 3)
plt.plot(t, true_trajectory[:, 2], 'k', label='True z')
plt.plot(t, state_estimate[:, 2], 'b', label='Estimated z')
plt.legend()

plt.tight_layout()
plt.show()

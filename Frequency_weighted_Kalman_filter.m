% 清空工作空间和命令窗口
clear;
clc;

% 1. 生成模拟数据

% 设置随机数生成器的种子
rng(0);

% 定义时间步长和总的时间步数
dt = 1.0;  
steps = 100; 

% 定义初始状态 [位置, 速度]
x = [0; 1]; 

% 定义状态转移矩阵
F = [1, dt; 0, 1];

% 定义过程噪声协方差
Q = [0.01, 0; 0, 0.01];

% 定义观测模型矩阵
H = [1, 0];

% 定义观测噪声协方差
R = 0.1;

% 生成真实状态和带噪声的观测数据
true_states = zeros(steps, 2);
noisy_measurements = zeros(steps, 1);

for k = 1:steps
    true_states(k, :) = x';
    x = F * x + mvnrnd([0, 0], Q)';
    z = H * x + sqrt(R) * randn();
    noisy_measurements(k) = z;
end

% 3. 应用卡尔曼滤波器

% 定义初始估计误差协方差
P0 = eye(2);

% 定义标准卡尔曼滤波器估计
standard_estimates = zeros(steps, 2);
x = true_states(1, :)';
P = P0;

for k = 1:steps
    [x, P] = kalmanFilter(F, H, Q, R, x, P, noisy_measurements(k));
    standard_estimates(k, :) = x';
end

% 定义频率加权卡尔曼滤波器估计
weighted_estimates = zeros(steps, 2);
R_weighted = R * 10;  % 增加观测噪声协方差
x = true_states(1, :)';
P = P0;

for k = 1:steps
    [x, P] = kalmanFilter(F, H, Q, R_weighted, x, P, noisy_measurements(k));
    weighted_estimates(k, :) = x';
end

% 4. 绘制结果

figure;
plot(true_states(:, 1), 'LineWidth', 1.5); hold on;
plot(noisy_measurements, '--', 'LineWidth', 1.5);
plot(standard_estimates(:, 1), 'LineWidth', 1.5);
plot(weighted_estimates(:, 1), '-.', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('位置');
title('卡尔曼滤波器估计');
legend('真实位置', '带噪声的观测', '标准卡尔曼滤波器估计', '频率加权卡尔曼滤波器估计');
grid on;

% 2. 定义卡尔曼滤波器函数

function [x, P] = kalmanFilter(F, H, Q, R, x, P, z)
    % 预测
    x = F * x;
    P = F * P * F' + Q;

    % 更新
    y = z - H * x;
    S = H * P * H' + R;
    K = P * H' / S;
    x = x + K * y;
    P = (eye(size(P)) - K * H) * P;
end

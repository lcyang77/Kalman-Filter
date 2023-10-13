% 定义参数
dt = 0.1; % 时间步长
N = 200; % 时间步数
sigma_a = 0.1; % 加速度标准差
sigma_z = 5; % 测量噪声标准差

% 定义状态转移矩阵
F = [1, dt; 0, 1];

% 定义过程噪声协方差矩阵
G = [0.5*dt^2; dt];
Q = G * G' * sigma_a^2;

% 定义测量矩阵
H = [1, 0];

% 定义测量噪声协方差矩阵
R = sigma_z^2;

% 初始状态
x0 = [0; 0];

% 生成真实状态和测量值
rng(0); % 设置随机数生成器的种子
a = randn(1, N) * sigma_a; % 生成加速度噪声
x_true = zeros(2, N);
x_true(:, 1) = x0;
for k = 2:N
    x_true(:, k) = F * x_true(:, k-1) + G * a(k-1);
end
z = H * x_true + randn(1, N) * sigma_z; % 生成测量值

% 执行卡尔曼滤波
x_est = zeros(2, N);
x_est(:, 1) = x0;
P = zeros(2, 2, N);
P(:, :, 1) = [0, 0; 0, 0]; % 初始协方差矩阵
for k = 2:N
    % 预测
    x_pred = F * x_est(:, k-1);
    P_pred = F * P(:, :, k-1) * F' + Q;
    
    % 更新
    y = z(:, k) - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_est(:, k) = x_pred + K * y;
    P(:, :, k) = P_pred - K * S * K';
end

% 绘制图像
figure;
plot(x_true(1, :), 'LineWidth', 1.5); hold on;
plot(z(1, :), 'o', 'MarkerSize', 4);
plot(x_est(1, :), 'LineWidth', 1.5);
xlabel('时间步');
ylabel('位置');
title('卡车位置的真实值、测量值和卡尔曼滤波估计值');
legend('真实位置', '测量位置', '卡尔曼滤波估计位置');
grid on;

% 没有设置随机种子，所以每次运行代码时都会生成不同的随机噪声。

% 清空变量和命令窗口
clear;
clc;

% 生成模拟数据
dt = 1; % 时间步长
steps = 100; % 时间步数
x = zeros(1, steps); % 真实状态
z = zeros(1, steps); % 测量数据
x(1) = 0; % 初始状态
Q = 0.01^2; % 系统噪声协方差
R = 0.1^2; % 测量噪声协方差

% 系统动态方程和测量方程
f = @(x) x + sin(x)*dt; % 非线性系统方程
h = @(x) x; % 测量方程

% 生成真实状态和测量数据
for t = 2:steps
    x(t) = f(x(t-1)) + sqrt(Q)*randn();
    z(t) = h(x(t)) + sqrt(R)*randn();
end

% 初始化 UKF 参数
nx = 1; % 状态维度
nz = 1; % 测量维度
alpha = 1e-3;
beta = 2;
kappa = 0;
lambda = alpha^2*(nx+kappa)-nx;
wm = [lambda/(nx+lambda) repmat(1/(2*(nx+lambda)), 1, 2*nx)]; % 权重
wc = wm;
wc(1) = wc(1) + (1 - alpha^2 + beta);
Xs = zeros(nx, 2*nx+1); % sigma 点
P = Q; % 初始协方差矩阵
xhat = zeros(1, steps); % 估计状态
xhat(1) = 0; % 初始状态估计

% UKF 滤波过程
for t = 2:steps
    % 计算 sigma 点
    X = sqrtm((nx+lambda)*P);
    Xs = [xhat(t-1) xhat(t-1)+X xhat(t-1)-X];
    
    % 预测
    Xs_pred = arrayfun(f, Xs);
    xhat_pred = wm*Xs_pred';
    P_pred = (wc .* (Xs_pred - xhat_pred)) * (Xs_pred - xhat_pred)';
    P_pred = P_pred + Q;
    
    % 更新
    Zs = arrayfun(h, Xs_pred);
    zhat = wm*Zs';
    Pzz = (wc .* (Zs - zhat)) * (Zs - zhat)';
    Pzz = Pzz + R;
    Pxz = (wc .* (Xs_pred - xhat_pred)) * (Zs - zhat)';
    K = Pxz / Pzz;
    xhat(t) = xhat_pred + K*(z(t) - zhat);
    P = P_pred - K*Pzz*K';
end

% 绘图
figure;
plot(1:steps, x, 'LineWidth', 1.5); hold on;
plot(1:steps, z, 'o', 'MarkerSize', 2);
plot(1:steps, xhat, 'LineWidth', 1.5);
legend('真实状态', '测量', 'UKF 估计');
grid on;
xlabel('时间步');
ylabel('状态');
title('无迹卡尔曼滤波器 (UKF) 状态估计');

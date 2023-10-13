% 定义时间步长和总的模拟时间
dt = 0.1;
T = 10;
steps = T / dt;

% 初始状态
x_true = [0; 0; pi/2];
x_est = x_true;

% 初始协方差矩阵
P = eye(3);

% 过程噪声协方差
Q = diag([0.1, 0.1, deg2rad(1.0)]).^2;

% 测量噪声协方差
R = diag([0.5, 0.5]).^2;

% 控制输入 [v, omega]
u = [1; 0.1];

% 用于存储真实轨迹和测量的变量
trajectory_true = zeros(steps, 3);
measurements = zeros(steps, 2);
estimates = zeros(steps, 3);

for k = 1:steps
    % 模拟真实状态
    x_true(1) = x_true(1) + dt * u(1) * cos(x_true(3));
    x_true(2) = x_true(2) + dt * u(1) * sin(x_true(3));
    x_true(3) = x_true(3) + dt * u(2);
    trajectory_true(k, :) = x_true';
    
    % 生成带噪声的测量
    z = [x_true(1); x_true(2)] + sqrt(R) * randn(2, 1);
    measurements(k, :) = z';
    
    % EKF 预测
    x_pred = x_est;
    x_pred(1) = x_pred(1) + dt * u(1) * cos(x_est(3));
    x_pred(2) = x_pred(2) + dt * u(1) * sin(x_est(3));
    x_pred(3) = x_pred(3) + dt * u(2);
    
    F = [1, 0, -dt * u(1) * sin(x_est(3));
         0, 1, dt * u(1) * cos(x_est(3));
         0, 0, 1];
    P_pred = F * P * F' + Q;
    
    % EKF 更新
    H = [1, 0, 0;
         0, 1, 0];
    K = P_pred * H' / (H * P_pred * H' + R);
    x_est = x_pred + K * (z - H * x_pred);
    P = (eye(3) - K * H) * P_pred;
    estimates(k, :) = x_est';
end

% 绘制结果
figure;
plot(trajectory_true(:, 1), trajectory_true(:, 2), 'b-', 'LineWidth', 2); hold on;
plot(measurements(:, 1), measurements(:, 2), 'ro', 'MarkerSize', 5);
plot(estimates(:, 1), estimates(:, 2), 'g-', 'LineWidth', 2);
legend('真实轨迹', '测量', 'EKF 估计');
xlabel('X');
ylabel('Y');
title('EKF 状态估计');
grid on;

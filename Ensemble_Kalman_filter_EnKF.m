
% 设置参数
sigma = 10; beta = 8/3; rho = 28;
initial_state = [1; 1; 1];
tspan = linspace(0, 10, 1000);
observation_noise_std = 1.0;
num_ensemble_members = 100;

% 生成洛伦茨系统的真实轨迹
[t, true_trajectory] = ode45(@(t, x) lorenz(t, x, sigma, beta, rho), tspan, initial_state);

% 添加噪声来模拟观测数据
observations = true_trajectory(:, 1:2) + observation_noise_std * randn(size(true_trajectory, 1), 2);

% 初始化集合
ensemble = initial_state + randn(3, num_ensemble_members);

% 集合卡尔曼滤波器来估计状态
state_estimate = zeros(length(tspan), 3);
for i = 2:length(tspan)
    dt = tspan(i) - tspan(i - 1);
    
    % 预测步骤
    for j = 1:num_ensemble_members
        [~, x_ensemble] = ode45(@(t, x) lorenz(t, x, sigma, beta, rho), [0 dt], ensemble(:, j));
        ensemble(:, j) = x_ensemble(end, :)';
    end
    
    % 更新步骤
    mean_ensemble = mean(ensemble, 2);
    Y = ensemble(1:2, :) + observation_noise_std * randn(2, num_ensemble_members);
    mean_Y = mean(Y, 2);
    Pxy = (ensemble - mean_ensemble) * (Y - mean_Y)' / (num_ensemble_members - 1);
    Pyy = cov(Y') + observation_noise_std^2 * eye(2);
    K = Pxy / Pyy;
    ensemble = ensemble + K * (observations(i, :)' - mean_Y);
    
    % 存储状态估计
    state_estimate(i, :) = mean(ensemble, 2)';
end

% 可视化结果
figure;

% x 状态
subplot(3, 1, 1);
plot(t, true_trajectory(:, 1), 'k', 'DisplayName', '真实 x');
hold on;
scatter(t, observations(:, 1), 5, 'r', 'DisplayName', '观测 x');
plot(t, state_estimate(:, 1), 'b', 'DisplayName', '估计 x');
hold off;
legend;
xlabel('时间');
ylabel('x 状态');

% y 状态
subplot(3, 1, 2);
plot(t, true_trajectory(:, 2), 'k', 'DisplayName', '真实 y');
hold on;
scatter(t, observations(:, 2), 5, 'r', 'DisplayName', '观测 y');
plot(t, state_estimate(:, 2), 'b', 'DisplayName', '估计 y');
hold off;
legend;
xlabel('时间');
ylabel('y 状态');

% z 状态
subplot(3, 1, 3);
plot(t, true_trajectory(:, 3), 'k', 'DisplayName', '真实 z');
hold on;
plot(t, state_estimate(:, 3), 'b', 'DisplayName', '估计 z');
hold off;
legend;
xlabel('时间');
ylabel('z 状态');

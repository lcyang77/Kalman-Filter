% 初始化
rng(0); % 设置随机种子
T = 50; % 时间步数
numParticles = 1000; % 粒子数量

real_x = zeros(1, T); % 真实状态
z = zeros(1, T); % 观测
estimates = zeros(1, T); % 估计

particles = randn(1, numParticles); % 初始化粒子
weights = ones(1, numParticles) / numParticles; % 初始化权重

% 生成真实状态和观测
for t = 2:T
    real_x(t) = real_x(t-1) + randn(); % 真实状态是一个随机漫步
    z(t) = real_x(t) + randn(); % 观测包含噪声
end

% 粒子滤波
for t = 1:T
    % 预测步骤
    particles = particles + randn(1, numParticles); % 添加过程噪声
    
    % 更新步骤
    innovation = z(t) - particles; % 计算创新（观测减去预测）
    likelihood = exp(-0.5 * innovation.^2); % 计算似然
    weights = weights .* likelihood; % 更新权重
    weights = weights / sum(weights); % 归一化权重
    
    % 状态估计
    estimates(t) = sum(particles .* weights); % 加权平均
    
    % 重采样
    indices = randsample(1:numParticles, numParticles, true, weights); % 基于权重进行重采样
    particles = particles(indices); % 更新粒子集
    weights = ones(1, numParticles) / numParticles; % 重置权重
end

% 绘图
figure;
hold on;
plot(1:T, real_x, 'b-', 'DisplayName', '真实状态'); % 真实状态
plot(1:T, z, 'k.', 'DisplayName', '观测'); % 观测
plot(1:T, estimates, 'r--', 'DisplayName', '估计'); % 粒子滤波器估计
hold off;
legend('show');
xlabel('时间步');
ylabel('状态');
title('使用粒子滤波器进行状态估计');
grid on;

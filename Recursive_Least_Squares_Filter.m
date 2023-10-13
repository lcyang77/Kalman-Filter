% 递归最小二乘(RLS)滤波器示例

% Step 1: 生成信号

N = 500; % 样本数量
t = linspace(0, 1, N); % 时间向量
freq = 5; % 正弦波频率
sine_wave = sin(2 * pi * freq * t); % 生成正弦波
noise = 0.5 * randn(1, N); % 生成随机噪声
signal = sine_wave + noise; % 得到含噪声的信号

% 定义RLS滤波器

filter_length = 4;
lambda_ = 0.99;
delta_ = 1.0;
weights = zeros(1, filter_length); % 初始化权重矩阵为零
P = delta_ * eye(filter_length); % 初始化误差协方差矩阵为delta倍的单位矩阵

estimated_signal = zeros(1, N);

for i = (filter_length+1):N
    x = signal(i-filter_length:i-1); % 获取当前输入
    d = signal(i); % 获取当前期望输出
    
    % 使用权重计算估计输出
    y = weights * x';
    
    % 更新误差协方差矩阵和权重
    k = (P * x') / (lambda_ + x * P * x');
    P = (P - k * x * P) / lambda_;
    weights = weights + k' * (d - y);
    
    estimated_signal(i) = y;
end

% 可视化结果

figure;
plot(t, signal, 'LineWidth', 1.2); hold on;
plot(t, estimated_signal, 'r', 'LineWidth', 1.2); hold off;
title('原始信号 vs. RLS估计');
legend('原始信号', 'RLS估计');
grid on;

% 递归最小二乘(RLS)滤波器示例

% Step 1: 生成信号

N = 500; % 样本数量
t = linspace(0, 1, N); % 时间向量
freq = 5; % 正弦波频率
sine_wave = sin(2 * pi * freq * t); % 生成正弦波
noise = 0.5 * randn(1, N); % 生成随机噪声
signal = sine_wave + noise; % 得到含噪声的信号

% 定义RLS滤波器

filter_length = 4;
lambda_ = 0.99;
delta_ = 1.0;
weights = zeros(1, filter_length); % 初始化权重矩阵为零
P = delta_ * eye(filter_length); % 初始化误差协方差矩阵为delta倍的单位矩阵

estimated_signal = zeros(1, N);

for i = (filter_length+1):N
    x = signal(i-filter_length:i-1); % 获取当前输入
    d = signal(i); % 获取当前期望输出
    
    % 使用权重计算估计输出
    y = weights * x';
    
    % 更新误差协方差矩阵和权重
    k = (P * x') / (lambda_ + x * P * x');
    P = (P - k * x * P) / lambda_;
    weights = weights + k' * (d - y);
    
    estimated_signal(i) = y;
end

% 可视化结果

figure;
plot(t, signal, 'LineWidth', 1.2); hold on;
plot(t, estimated_signal, 'r', 'LineWidth', 1.2); hold off;
title('原始信号 vs. RLS估计');
legend('原始信号', 'RLS估计');
grid on;


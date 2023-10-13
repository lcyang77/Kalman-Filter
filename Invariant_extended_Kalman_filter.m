clear; clc; close all;

% 定义非线性动力学模型
dynamicsModel = @(state, omega, dt) [state(1) + state(3)*cos(state(4))*dt; 
                                    state(2) + state(3)*sin(state(4))*dt; 
                                    state(3); 
                                    state(4) + omega*dt];

% 定义非线性测量模型
measurementModel = @(state, sensorPos) sqrt((state(1) - sensorPos(1))^2 + (state(2) - sensorPos(2))^2);

% 定义 IEKF 更新
IEKFUpdate = @(state, P, zMeas, sensorPos, Q, R, dt) iekfUpdate(state, P, zMeas, sensorPos, Q, R, dt, dynamicsModel, measurementModel);

% 模拟参数
dt = 0.1;  % 时间步长
steps = 200;  % 时间步数

% 初始化状态
trueState = [0.0; 0.0; 1.0; pi / 4]; 
estimatedState = [0.0; 0.0; 1.0; pi / 4];
P = eye(4);  % 协方差矩阵初始化
Q = eye(4) * 0.01;  % 过程噪声协方差
R = 0.1;  % 测量噪声协方差

% 传感器位置
sensorPos = [5.0, 5.0];

% 存储轨迹和测量值
trueTrajectory = zeros(2, steps);
estimatedTrajectory = zeros(2, steps);
measurements = zeros(1, steps);

% 模拟
for step = 1:steps
    omega = normrnd(0, 0.1);  % 生成角速度噪声
    w = normrnd(0, sqrt(R));  % 生成测量噪声

    trueState = dynamicsModel(trueState, omega, dt);  % 真实状态更新

    zTrue = measurementModel(trueState, sensorPos);  % 得到真实测量值
    zMeas = zTrue + w;  % 添加噪声

    [estimatedState, P] = IEKFUpdate(estimatedState, P, zMeas, sensorPos, Q, R, dt);  % IEKF 更新

    trueTrajectory(:, step) = trueState(1:2);  % 存储真实轨迹
    estimatedTrajectory(:, step) = estimatedState(1:2);  % 存储估计轨迹
    measurements(step) = zMeas;  % 存储测量值
end

% 绘图
figure;
plot(trueTrajectory(1, :), trueTrajectory(2, :), 'DisplayName', '真实轨迹');
hold on;
plot(estimatedTrajectory(1, :), estimatedTrajectory(2, :), 'DisplayName', '估计轨迹');
plot(sensorPos(1), sensorPos(2), 'ro', 'DisplayName', '传感器位置');
hold off;
legend;
xlabel('X 位置');
ylabel('Y 位置');
title('IEKF 非线性系统');
grid on;

% 定义 IEKF 更新的内部实现
function [updatedState, updatedP] = iekfUpdate(state, P, zMeas, sensorPos, Q, R, dt, dynamicsModel, measurementModel)
    for i = 1:2  % 进行两次迭代
        % 预测
        statePred = dynamicsModel(state, 0, dt);  % 零角速度预测

        % 计算雅可比矩阵
        xPred = statePred(1);
        yPred = statePred(2);
        Hx = (xPred - sensorPos(1)) / sqrt((xPred - sensorPos(1))^2 + (yPred - sensorPos(2))^2);
        Hy = (yPred - sensorPos(2)) / sqrt((xPred - sensorPos(1))^2 + (yPred - sensorPos(2))^2);
        H = [Hx, Hy, 0, 0];

        % 计算卡尔曼增益
        S = H * P * H' + R;
        K = P * H' / S;

        % 状态更新
        zPred = measurementModel(statePred, sensorPos);
        zDiff = zMeas - zPred;
        state = statePred + K * zDiff;

        % 协方差更新
        P = P - K * S * K';
    end
    updatedState = state;
    updatedP = P;
end

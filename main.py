from rl import ActorCritic,TimeSeriesEnv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("your_time_series_data.csv")
series = data["target_column"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series)

# 初始化 Actor-Critic 模型
# look_back: 状态历史窗口长度（输入特征维度）
# 1: 动作空间维度（输出预测值）
model = ActorCritic(look_back, 1)

# 使用 Adam 优化器分别优化 Actor 和 Critic 网络
actor_optimizer = optim.Adam(model.actor_mean.parameters(), lr=3e-4)  # Actor 学习率 3e-4
critic_optimizer = optim.Adam(model.critic.parameters(), lr=3e-4)    # Critic 学习率 3e-4

# 创建时间序列环境
# states: 输入状态（历史窗口数据），targets: 目标值（下一时间步的真实值）
env = TimeSeriesEnv(states, targets)

num_episodes = 100  # 训练总轮数
gamma = 0.99        # 折扣因子（未来奖励的重要性）
eps_clip = 0.2      # PPO 中的 clip 参数（限制策略更新幅度）

for episode in range(num_episodes):  # 开始训练循环
    state = env.reset()              # 重置环境，获取初始状态
    done = False                     # 标记是否完成当前 episode
    total_reward = 0                 # 累计当前 episode 的总奖励

    while not done:  # 时间步循环
        # 将当前状态转换为 PyTorch 张量，并增加批次维度 [1, look_back]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 通过模型获取动作（预测值）和动作分布（如高斯分布）
        action, dist = model.get_action(state_tensor)
        
        # 获取 Critic 对当前状态的价值估计
        value = model.get_value(state_tensor)

        # 在环境中执行动作，得到下一个状态、奖励和终止标志
        next_state, reward, done = env.step(action)

        # 计算下一个状态的价值（若存在）
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = model.get_value(next_state_tensor)
        else:
            next_value = torch.tensor([0])  # 若为终止状态，价值设为 0

        # 计算 TD 残差（delta）和优势值
        delta = reward + gamma * next_value - value  # TD-error
        advantage = delta.item()  # 单步优势值

        # Critic 更新：最小化 TD-error 的平方
        critic_loss = delta.pow(2).mean()  # Critic 损失（均方误差）
        critic_optimizer.zero_grad()       # 清空梯度
        critic_loss.backward()             # 反向传播计算梯度
        critic_optimizer.step()            # 更新 Critic 参数

        # Actor 更新：使用 PPO 的 clip 方法（Trusted Region Policy Optimization）
        # 计算新旧策略的概率比（ratio）
        ratio = torch.exp(dist.log_prob(action) - dist.log_prob(action).detach())  # 错误：此处 ratio 恒为 1！
        
        # 计算 Surrogate Loss（目标函数）
        surr1 = ratio * advantage                  # 未裁剪的目标
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage  # 裁剪后的目标
        actor_loss = -torch.min(surr1, surr2).mean()  # 最小化负的最小值（等效于最大化）

        # 更新 Actor 参数
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        total_reward += reward  # 累计奖励
        state = next_state      # 更新状态

    # 打印训练信息
    print(f"Episode [{episode+1}/{num_episodes}], Total Reward: {total_reward:.4f}")
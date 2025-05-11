import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_states(data, look_back=1):
    """
    将时间序列数据转换为监督学习格式（输入特征 X 和目标变量 Y）。
    
    Args:
        data (np.ndarray): 输入的时间序列数据，形状为 [n_samples, n_features]。
        look_back (int): 使用过去多少个时间步的数据作为输入特征（历史窗口大小）。
    
    Returns:
        tuple: (X, Y)
            - X: 输入特征，形状为 [n_samples - look_back, look_back]
            - Y: 目标变量，形状为 [n_samples - look_back, ]
    """
    X, Y = [], []  # 初始化输入特征列表和目标变量列表
    
    # 遍历时间序列数据，构建输入-目标对
    for i in range(len(data) - look_back):
        # 提取历史窗口数据（从 i 到 i+look_back 的时间步），取第 0 列作为特征
        X.append(data[i:(i + look_back), 0])
        
        # 提取目标值（下一个时间步的数据），取第 0 列
        Y.append(data[i + look_back, 0])
    
    # 将列表转换为 NumPy 数组并返回
    return np.array(X), np.array(Y)


class TimeSeriesEnv:
    def __init__(self, states, targets):
        """
        初始化时间序列环境。
        
        Args:
            states (np.ndarray): 输入的观测状态（时间序列的历史窗口），形状为 [n_samples, look_back]
            targets (np.ndarray): 目标值（时间序列的未来值），形状为 [n_samples, ]
        """
        self.states = states              # 存储所有观测状态（历史时间窗口）
        self.targets = targets            # 存储所有目标值（未来值）
        self.current_step = 0             # 当前时间步，初始化为0
        self.n_steps = len(states)        # 总时间步数（等于 states 的长度）

    def reset(self):
        """
        重置环境到初始状态。
        
        Returns:
            np.ndarray: 初始状态（第一个时间步的观测）
        """
        self.current_step = 0             # 重置当前时间步为0
        return self.states[self.current_step]  # 返回第一个时间步的状态

    def step(self, action):
        """
        执行一步动作，返回下一个状态、奖励和是否结束。
        
        Args:
            action (torch.Tensor): 智能体采取的动作（预测值），形状为 [1, 1]
        
        Returns:
            tuple: 
                - next_state (np.ndarray or None): 下一个状态（若未结束）或 None
                - reward (float): 负的预测误差（越接近真实值，奖励越高）
                - done (bool): 是否达到序列末尾
        """
        self.current_step += 1  # 更新时间步
        
        # 判断是否结束：当 current_step >= n_steps - 1 时结束
        done = self.current_step >= self.n_steps - 1
        
        # 获取下一个状态（若未结束）或 None（结束时）
        next_state = self.states[self.current_step] if not done else None
        
        # 计算奖励：负的预测误差（预测值越接近真实值，奖励越高）
        reward = -abs(action.item() - self.targets[self.current_step])  # action.item() 将张量转为浮点数
        
        return next_state, reward, done  # 返回下一个状态、奖励、是否结束

class ActorCritic(nn.Module):
    """
    Actor-Critic 模型：用于强化学习中的策略和价值网络联合建模。
    
    Attributes:
        obs_dim (int): 观测空间（状态）的维度。
        action_dim (int): 动作空间的维度。
        feature_layer (nn.Sequential): 共享的特征提取网络。
        actor_mean (nn.Linear): Actor 网络的输出层，预测动作均值。
        actor_log_std (nn.Parameter): Actor 网络的动作对数标准差。
        critic (nn.Linear): Critic 网络，输出状态价值。
    """
    
    def __init__(self, obs_dim, action_dim):
        """
        初始化 Actor-Critic 模型。
        
        Args:
            obs_dim (int): 观测空间（状态）的维度。
            action_dim (int): 动作空间的维度。
        """
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 共享特征提取层：将输入观测转换为高维特征表示
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 64),  # 输入层：obs_dim -> 64
            nn.ReLU(),                 # 激活函数
            nn.Linear(64, 64),         # 隐藏层：64 -> 64
            nn.ReLU()                  # 激活函数
        )
        
        # Actor 网络：输出动作的均值和标准差
        self.actor_mean = nn.Linear(64, action_dim)  # 预测动作均值
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))  # 动作的对数标准差（可学习参数）
        
        # Critic 网络：输出状态价值估计
        self.critic = nn.Linear(64, 1)  # 状态价值估计

    def forward(self, x):
        """
        前向传播：提取输入观测的共享特征。
        
        Args:
            x (torch.Tensor): 输入观测张量，形状为 [batch_size, obs_dim]
        
        Returns:
            torch.Tensor: 提取的特征，形状为 [batch_size, 64]
        """
        features = self.feature_layer(x)
        return features

    def get_action(self, x):
        """
        根据当前策略网络生成动作样本。
        
        Args:
            x (torch.Tensor): 输入观测张量，形状为 [batch_size, obs_dim]
        
        Returns:
            tuple: 
                - action (torch.Tensor): 采样的动作，形状为 [batch_size, action_dim]
                - dist (Normal): 高斯分布对象
        """
        features = self.forward(x)  # 提取特征
        mean = self.actor_mean(features)  # 计算动作均值
        log_std = self.actor_log_std.expand_as(mean)  # 扩展 log_std 到均值的形状
        std = torch.exp(log_std)  # 计算标准差
        dist = Normal(mean, std)  # 构造高斯分布
        action = dist.sample()  # 从分布中采样动作
        return action, dist

    def get_value(self, x):
        """
        获取状态价值估计。
        
        Args:
            x (torch.Tensor): 输入观测张量，形状为 [batch_size, obs_dim]
        
        Returns:
            torch.Tensor: 状态价值，形状为 [batch_size, 1]
        """
        features = self.forward(x)  # 提取特征
        value = self.critic(features)  # 计算状态价值
        return value
    
def predict(model, env):
    """
    使用训练好的模型在给定的环境中进行时间序列预测。
    
    Args:
        model: 训练好的模型（如深度强化学习模型）。
        env: 时间序列环境（如 TimeSeriesEnv 类实例）。
    
    Returns:
        predictions: 反归一化后的预测值。
        true_values: 反归一化后的真实值。
    """
    model.eval()  # 将模型设置为评估模式（关闭 dropout、batch norm 等训练专用操作）
    
    predictions = []  # 存储模型预测值
    true_values = []  # 存储真实目标值
    
    state = env.reset()  # 重置环境，获取初始状态（第一个时间步的观测）
    done = False  # 标记是否结束预测
    
    while not done:
        # 将当前状态转换为 PyTorch 张量，并增加批次维度（形状变为 [1, look_back]）
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 使用模型生成动作（预测值），`_` 表示忽略其他输出（如状态价值）
        action, _ = model.get_action(state_tensor)
        
        # 保存预测值
        predictions.append(action.item())  # 将张量转为浮点数并存入列表
        
        # 保存当前时间步的真实目标值
        true_values.append(env.targets[env.current_step])
        
        # 在环境中执行动作，获取下一个状态、奖励（未使用）和终止标志
        next_state, _, done = env.step(action)
        
        # 更新状态
        state = next_state
    
    # 反归一化：将预测值和真实值从归一化范围（如 [0,1]）恢复到原始数据范围
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1))
    
    return predictions, true_values  # 返回预测值和真实值
import numpy as np
import pandas as pd
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesEnv:
    def __init__(self, dataloader):
        """
        初始化时间序列环境，支持批次数据。
        
        Args:
            dataloader (DataLoader): 包含时间序列数据的批次加载器。
        """
        self.dataloader = dataloader
        self.data_iter = None
        self.current_batch = None
        self.batch_index = 0
        self.current_step = 0

    def reset(self):
        """
        重置环境到初始状态，并返回第一个批次的第一个状态。
        """
        self.data_iter = iter(self.dataloader)
        self.current_step = 0
        self.current_batch = next(self.data_iter)
        self.batch_index = 0
        return self.current_batch

    def step(self, action):
        """
        执行一步动作，返回下一个状态、奖励和是否结束。
        
        Args:
            action (torch.Tensor): 智能体采取的动作（预测值）

        Returns:
            tuple: 下一个状态、奖励、是否结束
        """
        target = self.current_batch[1][self.batch_index].item()
        reward = -abs(action.item() - target)
        self.batch_index += 1
        self.current_step += 1

        # 检查是否还有更多样本在当前批次中
        if self.batch_index < len(self.current_batch[0]):
            next_state = self.current_batch[0][self.batch_index]
            done = False
        else:
            try:
                self.current_batch = next(self.data_iter)
                self.batch_index = 0
                next_state = self.current_batch[0][self.batch_index]
                done = False
            except StopIteration:
                next_state = None
                done = True

        return next_state, reward, done

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
        # self.critic = nn.Linear(64, 1)  # 状态价值估计

        self.hoding_money = 1
        self.hoding_cost = 0 
        self.hoding_share = 0

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

    def cal_benefit(self,current_price):
        # 计算每股的成本价
        cost_price = self.hoding_cost / self.hoding_share
        
        # 计算浮盈
        floating_profit = (current_price - cost_price) * self.hoding_share

        return floating_profit

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
        # 计算浮盈
        # features = self.forward(x)  # 提取特征
        value = self.cal_benefit(x[0])
        return value

    def load_weight(self,ckpt):
        self.load_state_dict(ckpt['AC_model_state_dict'])
    
    def save_weight(self,ckpt):
        ckpt['AC_model_state_dict'] = self.state_dict()
        return ckpt
    
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
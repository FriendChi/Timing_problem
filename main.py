from lib.rl import *
from lib.arguments import *
from lib.data import *
from lib.utils import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler

def train_model(model,
                env,
                actor_optimizer,
                actor_scheduler,
                critic_optimizer,
                critic_scheduler,
                args,
                writer):
    """
    训练模型并返回每个 episode 的总奖励列表。

    Args:
        model (nn.Module): Actor-Critic 模型。
        env: OpenAI Gym 或自定义环境实例。
        actor_optimizer (torch.optim.Optimizer): Actor 优化器。
        critic_optimizer (torch.optim.Optimizer): Critic 优化器。
        args (dict): 各种超参数字典，需包含：
            - num_episodes (int): 训练 episode 数量
            - gamma (float): 折扣因子
            - eps_clip (float): PPO 裁剪阈值
            - print_interval (int): 每隔多少个 episode 打印一次日志

    Returns:
        rewards_list (list of float): 每个 episode 的累计奖励
    """
    rewards_list = []

    # 外层进度条：episodes
    for episode in trange(args['num_episodes'], desc='Episodes', unit='epi'):
        state_tensor = env.reset()
        done = False
        total_reward = 0.0

        # 如果想要显示每个 timestep，可启用下面的 tqdm
        # step_iter = tqdm(desc='Steps', leave=False, unit='step')

        #智能体与环境连续交互。只要当前的状态不是终止状态
        while not done:
            # step_iter.update(1)

            # 状态转 tensor “状态”可以包括市场价格、交易量、技术指标（如移动平均线）、宏观经济数据等信息
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Actor-Critic 前向
            action, dist = model.get_action(state_tensor[0]) # 当前状态下采取的动作 action 和该动作的概率分布 dist
            value = model.get_value(state_tensor[1]) # 计算当前状态的价值估计 value，这里的“价值”可以理解为预期收益。

            # 环境交互
            next_state, reward, done = env.step(action.numpy())  # 根据动作与环境交互，得到下一状态next_state、即时奖励reward以及是否结束标志 done

            # 下一个状态价值
            if not done: # 如果当前不是终止状态，则计算下一个状态的价值估计 next_value。
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                next_value = model.get_value(next_state_tensor)
            else: # 如果是终止状态，则将 next_value 设定为 0，因为没有后续状态需要考虑。
                next_value = torch.tensor([0.0])

            # 计算 TD-error 和优势
            delta = reward + args['gamma'] * next_value - value # 实际奖励加上未来折扣后的价值预测与当前价值估计之间的差异
            advantage = delta.item()

            # Critic 更新
            critic_loss = delta.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor 更新 (PPO clip)
            ratio = torch.exp(dist.log_prob(action) - dist.log_prob(action).detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - args['eps_clip'], 1 + args['eps_clip']) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            total_reward += reward
            state = next_state

        # step_iter.close()

        rewards_list.append(total_reward)

        # 定期输出日志
        writer.add_scalar('Train', total_reward, episode)
        actor_scheduler.step()
        critic_scheduler.step()
        if (episode + 1) % args.get('print_interval', 10) == 0:
            # 较高的平均奖励通常意味着智能体正在学习如何更有效地执行任务以获得更高的奖励。
            avg_rew = sum(rewards_list[-args['print_interval']:]) / args['print_interval']
            print(f"[Episode {episode+1:4d}/{args['num_episodes']}] "
                  f"Avg Reward (last {args['print_interval']}): {avg_rew:.3f}")
            save_checkpoint(model, 
                    actor_optimizer, 
                    critic_optimizer,
                    actor_scheduler,
                    critic_scheduler,
                    episode,
                    filename=args['ckpt_dir'])
            

    print("Training complete.")
    return rewards_list


if __name__ == '__main__':
    # 获取指令
    parser = get_parser()
    args = vars(parser.parse_args())
    print(args)

    # 获取当前时间并格式化为字符串，精确到秒
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建 SummaryWriter，日志保存在以当前时间命名的文件夹中
    writer = SummaryWriter(f'runs/{args["task_name"]}/{current_time}')

    # 数据处理
    data = pd.read_csv(args['csv_dir'])
    dataset = TimeSeriesDataset(data, look_back=args['look_back'])
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)

    # 初始化 Actor-Critic 模型
    # look_back: 状态历史窗口长度（输入特征维度）
    # 1: 动作空间维度（输出预测值）
    model = ActorCritic(dataset.input_dim, 1)

    # 使用 Adam 优化器分别优化 Actor 和 Critic 网络
    actor_optimizer = optim.Adam(model.actor_mean.parameters(), lr=args['a_lr'])  # Actor 学习率 3e-4
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=args['c_lr'])    # Critic 学习率 3e-4

    actor_scheduler = lr_scheduler.StepLR(
        actor_optimizer,
        step_size=30,
        gamma=0.1
    )
    critic_scheduler = lr_scheduler.StepLR(
        critic_optimizer,
        step_size=30,
        gamma=0.1
    )

    # 创建时间序列环境
    env = TimeSeriesEnv(dataloader)

    args['print_interval'] = int(args['num_episodes']*0.1)

    # 训练模型
    rewards_list = train_model(model,
                                env,
                                actor_optimizer,
                                actor_scheduler,
                                critic_optimizer,
                                critic_scheduler,
                                args,
                                writer)

    writer.close()

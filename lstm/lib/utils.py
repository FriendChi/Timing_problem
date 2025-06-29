import torch


def save_checkpoint(model, 
                    actor_optimizer, 
                    critic_optimizer,
                    actor_scheduler,
                    critic_scheduler,
                    episode,
                    filename=None):
    """
    保存模型、优化器和学习率调度器的状态。
    
    Args:
        model (nn.Module): 要保存的模型（如 ActorCritic）
        actor_optimizer (torch.optim.Optimizer): Actor 的优化器
        critic_optimizer (torch.optim.Optimizer): Critic 的优化器
        actor_scheduler (torch.optim.lr_scheduler): Actor 的学习率调度器
        critic_scheduler (torch.optim.lr_scheduler): Critic 的学习率调度器
        filename (str): 保存路径
    """
    if filename is None:
        return
    checkpoint = {
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'actor_scheduler_state_dict': actor_scheduler.state_dict(),
        'critic_scheduler_state_dict': critic_scheduler.state_dict(),
        'episode':episode,
    }
    checkpoint = model.save_weight(checkpoint)

    torch.save(checkpoint, filename)
    print(f"Model and optimizers saved to {filename}")


# 加载模型和优化器
def load_checkpoint(model, 
                    actor_optimizer, 
                    critic_optimizer,
                    actor_scheduler,
                    critic_scheduler,
                    filename='checkpoint.pth'):
    """
    加载模型、优化器和学习率调度器的状态。
    
    Args:
        model (nn.Module): 模型对象
        actor_optimizer (torch.optim.Optimizer): Actor 的优化器
        critic_optimizer (torch.optim.Optimizer): Critic 的优化器
        actor_scheduler (torch.optim.lr_scheduler): Actor 的学习率调度器
        critic_scheduler (torch.optim.lr_scheduler): Critic 的学习率调度器
        filename (str): 文件路径
    """
    checkpoint = torch.load(filename)

    model.load_weight(checkpoint)
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
    critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

    print(f"Model and optimizers loaded from {filename}")
    return model, actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler
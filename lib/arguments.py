import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv_dir',
        action='store',
        type=str,
        help='csv位置',
        required=True
    )
    parser.add_argument(
        '--look_back',
        action='store',
        type=int,
        help='时间窗口',
        default=1,
    )
    parser.add_argument(
        '--a_lr',
        action='store',
        type=float,
        help='演员学习率',
        default=3e-4,
    )
    parser.add_argument(
        '--c_lr',
        action='store',
        help='批评家学习率',
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        '--batch_size',
        action='store',
        type=int,
        help='批次',
        default=1,
    )
    parser.add_argument(
        '--num_episodes',
        action='store',
        type=int,
        help='训练总轮数',
        default=100,
    )
    parser.add_argument(
        '--gamma',
        action='store',
        help='折扣因子（未来奖励的重要性）',
        type=float,
        default=0.99,
    )
    parser.add_argument(
        '--eps_clip',
        action='store',
        help='PPO 中的 clip 参数（限制策略更新幅度）',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--task_name',
        action='store',
        type=str,
        help='任务名称，决定了记录文件对应的文件夹',
        default='default',
    )
    parser.add_argument(
        '--ckpt_dir',
        action='store',
        type=str,
        help='指向保存ckpt的文件夹',
        default=None,
    )

    return parser
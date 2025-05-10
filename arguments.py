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
        default=10,
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

    return parser
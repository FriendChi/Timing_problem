import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 定义 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, look_back=1, target_col=0):
        self.data = data
        self.look_back = look_back
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.look_back, 0]  # 取第 0 列作为输入特征
        y = self.data[idx + self.look_back, self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == '__main__':
    # 示例数据
    data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)  # 单列时间序列
    look_back = 10

    # 创建 Dataset 和 DataLoader
    dataset = TimeSeriesDataset(data, look_back=look_back)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 使用 DataLoader
    for batch_x, batch_y in dataloader:
        print("Batch X:", batch_x.shape)  # [batch_size, look_back]
        print("Batch Y:", batch_y.shape)  # [batch_size]
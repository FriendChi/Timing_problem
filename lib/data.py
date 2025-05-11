import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 定义 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, look_back=1, target_feature='Close'):
        """
        初始化时间序列数据集，确保输入为 pandas DataFrame，
        并在归一化前保留 DataFrame 格式，归一化后转换为 numpy 数组。
        
        Args:
            data (pd.DataFrame): 时间序列数据（必须为 DataFrame）
            look_back (int): 历史窗口长度
            target_col (str): 目标列特征名
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        # 1. 归一化前保留为 DataFrame 格式
        self.original_df = data.copy()

        # 2. 执行归一化处理，返回归一化后的 DataFrame
        normalized_df = self.__normalize(self.original_df)

        # 3. 检查目标列是否存在
        if target_feature not in normalized_df.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame columns.")

        # 4. 获取目标列的索引
        target_col = normalized_df.columns.get_loc(target_feature)

        # 3. 归一化后转换为 numpy 数组
        self.data = normalized_df.values

        # 4. 保存参数
        self.look_back = look_back
        self.target_col = target_col

    def __normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对除第一列外的所有列进行 Min-Max 归一化，除非第一列列名包含 'date'（不区分大小写），
        在这种情况下，第一列不参与归一化。

        Args:
            df (pd.DataFrame): 输入 DataFrame

        Returns:
            pd.DataFrame: 归一化后的 DataFrame
        """
        # 获取第一列的列名
        first_col_name = df.columns[0]

        # 判断列名是否包含 'date'（不区分大小写）
        if 'date' in str(first_col_name).lower():
            # 分离第一列和其他列
            other_cols = df.iloc[:, 1:]

            # 对其他列进行 Min-Max 归一化
            if not other_cols.empty:
                min_vals = other_cols.min()
                max_vals = other_cols.max()
                normalized_other = (other_cols - min_vals) / (max_vals - min_vals + 1e-8)
            else:
                normalized_other = other_cols

            # 不合并
            normalized_df = normalized_other

            # # 合并归一化后的列与第一列
            # normalized_df = pd.concat([first_col, normalized_other], axis=1)

        else:
            # 对所有列进行归一化
            min_vals = df.min()
            max_vals = df.max()
            normalized_df = (df - min_vals) / (max_vals - min_vals + 1e-8)

        return normalized_df

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, idx):
        # 使用NumPy数组进行切片操作
        x = self.data[idx:idx + self.look_back, 0]  # 假设输入特征总是第0列
        y = self.data[idx + self.look_back, self.target_col]
        print()
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
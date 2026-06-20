import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: numpy array, 形状为 (总天数, 特征数)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 输入窗口：[index : index + seq_len]
        x = self.data[index : index + self.seq_len]
        # 预测窗口：[index + seq_len : index + seq_len + pred_len]
        y = self.data[index + self.seq_len : index + self.seq_len + self.pred_len]
        return x, y

# 生成模拟的股票数据：2000天，5个特征（开高低收、成交量）
np.random.seed(42)
num_days = 2000
num_features = 5

# 模拟一个带随机游走（类似股价）的数据
simulated_stock_data = np.cumsum(np.random.randn(num_days, num_features), axis=0) + 100

# 划分训练集和测试集
train_data = simulated_stock_data[:1600]
test_data = simulated_stock_data[1600:]

# 设定窗口参数
SEQ_LEN = 30   # 用过去 30 天的数据
PRED_LEN = 5   # 预测未来 5 天

train_dataset = StockDataset(train_data, SEQ_LEN, PRED_LEN)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

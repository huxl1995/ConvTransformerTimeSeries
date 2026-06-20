import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


# =====================================================================
# 1. 步骤：定义位置编码 (Positional Encoding)
# =====================================================================
class PositionalEncoding(nn.Module):
    """
    为时间序列提供绝对时间位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


# =====================================================================
# 2. 步骤：定义多维时间序列 Transformer Encoder 模型
# =====================================================================
class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seq_len: int, pred_len: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        """
        参数说明:
        - input_dim: 输入的多维特征数 (如 3 维序列)
        - output_dim: 输出的预测特征数 (通常等于 input_dim)
        - seq_len: 输入的时间步长 (历史窗口长度)
        - pred_len: 预测的时间步长 (未来预测长度)
        - d_model: Transformer 内部的特征维度
        - nhead: 多头注意力机制的头数 (需能被 d_model 整除)
        - num_layers: Encoder 堆叠层数
        - dim_feedforward: 前沿全连接层的维度
        """
        super(TimeSeriesTransformerEncoder, self).__init__()

        # 线性投射层：将原始输入的多维特征映射到 d_model 维度
        self.input_linear = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # 定义 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 保持 [batch, seq, feature] 的经典结构
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：将编码后的 [batch_size, seq_len, d_model] 映射到预测的未来时序数据
        # 这里使用展平后映射到整个未来时间窗口，实现多步直接预测
        self.output_linear = nn.Linear(seq_len * d_model, pred_len * output_dim)

        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 1. src shape: [batch_size, seq_len, input_dim]
        x = self.input_linear(src)  # [batch_size, seq_len, d_model]

        # 2. 加入位置编码
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]

        # 3. 通过 Transformer Encoder 计算注意力
        memory = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # 4. 展平并映射到未来输出空间
        batch_size = memory.size(0)
        memory_flattened = memory.reshape(batch_size, -1)  # [batch_size, seq_len * d_model]

        out = self.output_linear(memory_flattened)  # [batch_size, pred_len * output_dim]

        # 5. 重新调整形状为 [batch_size, pred_len, output_dim]
        out = out.reshape(batch_size, self.pred_len, self.output_dim)
        return out


# =====================================================================
# 3. 步骤：使用示例与模拟数据训练
# =====================================================================
if __name__ == "__main__":
    # 模拟超参数设置
    batch_size = 16
    seq_len = 24  # 历史输入 24 个时间步 (例如过去 24 小时)
    pred_len = 6  # 预测未来 6 个时间步 (例如未来 6 小时)
    num_features = 3  # 3维多维时间序列 (例如：温度、湿度、风速)

    # 1. 生成模拟的时间序列训练数据 (这里用正弦/余弦模拟多维趋势)
    np.random.seed(42)
    torch.manual_seed(42)

    # 虚构一条超长的时间序列
    total_len = 1000
    t = np.linspace(0, 50, total_len)
    f1 = np.sin(t)
    f2 = np.cos(t * 1.5)
    f3 = np.sin(t * 2.0) + np.cos(t)
    full_data = np.stack([f1, f2, f3], axis=1)  # [total_len, 3]

    # 转换为滑动窗口数据集
    X_list, Y_list = [], []
    for i in range(total_len - seq_len - pred_len + 1):
        X_list.append(full_data[i: i + seq_len])
        Y_list.append(full_data[i + seq_len: i + seq_len + pred_len])

    X_train = torch.tensor(np.array(X_list), dtype=torch.float32)  # [N, seq_len, 3]
    Y_train = torch.tensor(np.array(Y_list), dtype=torch.float32)  # [N, pred_len, 3]

    print(f"训练集输入尺寸: {X_train.shape}")
    print(f"训练集标签尺寸: {Y_train.shape}")

    # 2. 实例化模型
    model = TimeSeriesTransformerEncoder(
        input_dim=num_features,
        output_dim=num_features,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1
    )

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 简易训练循环 (10 个 Epoch 演示)
    model.train()
    epochs = 10
    dataset_size = X_train.size(0)

    print("\n--- 开始训练 ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # 打乱数据
        permutation = torch.randperm(dataset_size)

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]

            optimizer.zero_grad()
            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        print(f"Epoch [{epoch + 1}/{epochs}], MSE Loss: {epoch_loss / dataset_size:.4f}")

    # =====================================================================
    # 4. 步骤：测试与模型推理演示
    # =====================================================================
    model.eval()
    with torch.no_grad():
        # 抽取一条测试数据 (假设是当前最新捕获的 24 步历史数据)
        sample_input = X_train[0:1]  # Shape: [1, 24, 3]
        sample_target = Y_train[0:1]  # Shape: [1, 6, 3]

        # 预测未来 6 步
        sample_prediction = model(sample_input)  # Shape: [1, 6, 3]

        print("\n--- 推理结果演示 ---")
        print("输入数据(最后一步):", sample_input[0, -1, :].numpy())
        print("预测未来第一步:   ", sample_prediction[0, 0, :].numpy())
        print("实际未来第一步:   ", sample_target[0, 0, :].numpy())

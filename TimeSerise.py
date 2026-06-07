import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, output_len):
        super(TimeSeriesTransformer, self).__init__()
        # 1. 特征映射：将多维输入映射到 d_model
        self.feature_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))

        # 2. Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 3. 预测头：映射回多维输出
        self.out_projection = nn.Linear(d_model, num_features * output_len)
        self.output_len = output_len
        self.num_features = num_features

    def forward(self, src):
        # src shape: (Batch, Seq_len, Features)
        batch_size = src.size(0)

        # 投影与位置编码
        x = self.feature_projection(src)
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer 编码器需要 (Seq_len, Batch, Features) 格式
        x = x.permute(1, 0, 2)
        memory = self.transformer_encoder(x)

        # 取最后一个时间步的输出或全局池化，这里以最后一个时间步为例
        last_hidden = memory[-1, :, :]

        # 输出预测值
        output = self.out_projection(last_hidden)
        # 重塑为 (Batch, Output_len, Features)
        output = output.reshape(batch_size, self.output_len, self.num_features)
        return output

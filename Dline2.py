from Dline import *
from Dline1 import *

# 1. 实例化模型、损失函数和优化器
model = DLinearForStock(seq_len=SEQ_LEN, pred_len=PRED_LEN, num_features=num_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. 训练循环
epochs = 10
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        # batch_x: [32, 30, 5], batch_y: [32, 5, 5]
        optimizer.zero_grad()

        # 预测
        outputs = model(batch_x)

        # 计算损失
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# 3. 模拟单次预测验证
model.eval()
with torch.no_grad():
    # 模拟最近30天的股票真实数据
    recent_30_days = torch.tensor(test_data[:SEQ_LEN], dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度 -> [1, 30, 5]

    # 预测未来5天
    future_5_days_pred = model(recent_30_days)

    print("\n--- 预测完成 ---")
    print("输入最近1天的价格(最后一行):", recent_30_days[0, -1, :].numpy())
    print("预测未来5天的价格:\n", future_5_days_pred[0].numpy())


from trans import *
from data import *
def loadData(path):
    return np.loadtxt(path, delimiter=",", skiprows=1,dtype=np.float64)
if __name__=="__main__":
    oriPath="all.csv"
    dstPath="data1.csv"
    convertData(oriPath,dstPath)
    data=loadData(dstPath)
    xTrain=[]
    yTrain=[]
    for i in range(len(data)-31):
        xTrain.append(data[i:i+30])
        yTrain.append(data[i+31:i+32])
    X_train = torch.tensor(np.array(xTrain), dtype=torch.float32)  # [N, seq_len, 3]
    Y_train = torch.tensor(np.array(yTrain), dtype=torch.float32)  # [N, pred_len, 3]

    print(f"训练集输入尺寸: {X_train.shape}")
    print(f"训练集标签尺寸: {Y_train.shape}")
    # 模拟超参数设置
    batch_size = 16
    seq_len = 30  # 历史输入 24 个时间步 (例如过去 24 小时)
    pred_len = 1  # 预测未来 6 个时间步 (例如未来 6 小时)
    num_features = 4  # 3维多维时间序列 (例如：温度、湿度、风速)
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
    epochs = 100
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

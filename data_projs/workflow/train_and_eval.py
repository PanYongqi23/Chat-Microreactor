from parse_json import *
from model import Net
import torch
from torch import nn, optim
from model import Net
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

Raw_PATH = r".\data\temp_results"
Processed_PATH = r".\data\processed"
Model_PATH = r".\data\model"
DEVICE = torch.device("cuda")

# Create the directories if they do not exist
for path in [Model_PATH, Processed_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

#用temp_results里面的东西把模型训练几轮，更新权重，然后清空。
def incremental_training(processed_path=Processed_PATH, model_path=Model_PATH+'\\Net.pth', num_epochs=35, batch_size=64):
    global DIM
    
    data = parse_path(Raw_PATH) #处理并存储处理后的pth数据
    
    # 加载包含数据的.pth文件
    #data = torch.load(processed_path+)
    
    # 假设data是一个字典，包含'embeddings'和'labels'键
    inputs = data['embeddings'].to(DEVICE)
    targets = data['labels'].float().to(DEVICE)

    # 创建TensorDataset
    dataset = TensorDataset(inputs, targets)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 获取embedding的大小，即inputs的第二个维度
    DIM = inputs.shape[1]
    print("dim：",DIM)
    # 加载模型结构
    model = Net(DIM).to(DEVICE)

    # 加载之前训练的权重（如果存在）
    '''
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model weights")
    else:
        print("No existing model found, starting training from scratch")
    '''
    
    # 设置损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 确保这与您的任务相匹配
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # 将模型设置为训练模式
    model.train()

    # 训练模型多次
    for epoch in range(num_epochs):
        for inputs_batch, targets_batch in dataloader:
            # 清零优化器的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs_batch)

            # 计算损失
            loss = criterion(outputs, targets_batch)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

    # 保存训练后的模型权重
    torch.save(model.state_dict(), model_path)
    print("Model weights saved to", model_path)

# 使用示例
# incremental_training('path/to/processed_data.pth', 'path/to/save_model.pth', num_epochs=10)

def predict_positive_probability_glm(embedding, model_path=Model_PATH+'\\Net.pth', device='cuda'):
    global DIM
    
    # 确保模型权重存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    # 加载模型结构
    model = Net(DIM).to(device)

    def check_model():
        # 检查模型的第一层（通常是全连接层或卷积层）的输入特征数
        first_layer = next(model.children())
        if hasattr(first_layer, 'in_features'):
            print(f"First layer in_features: {first_layer.in_features}")
        elif hasattr(first_layer, 'in_channels'):
            print(f"First layer in_channels: {first_layer.in_channels}")

        # 检查 embedding 的维度
        print(f"Embedding shape: {embedding.shape}")

        # 如果上述检查一切正常，接下来检查 BatchNorm 层的 num_features
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                print(f"BatchNorm layer {name} num_features: {layer.num_features}")
    
    #check_model()

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 将模型设置为评估模式

    # 检查 embedding 是否是 NumPy 数组，如果是，则转换为 PyTorch 张量
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)

    # 确保embedding是正确的设备上的tensor
    embedding = embedding.to(device)

    # 检查并转换输入的dtype为float
    if embedding.dtype != torch.float32:
        embedding = embedding.float()

    # 检查输入数据的维度，确保它与模型期望的输入维度相匹配
    # 如果模型期望二维输入，则重塑embedding
    if embedding.dim() == 3 and embedding.size(1) == 1:
        embedding = embedding.squeeze(1)  # 移除大小为1的维度

    # 如果只有一个样本，增加一个批次维度
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    # 使用模型进行预测
    with torch.no_grad():  # 确保不计算梯度
        output = model(embedding)

    # 应用sigmoid函数来获取阳性概率
    probabilities = torch.sigmoid(output)

    # 获取阳性概率
    positive_probability = probabilities.item() if probabilities.dim() == 0 else probabilities.mean().item()

    return positive_probability

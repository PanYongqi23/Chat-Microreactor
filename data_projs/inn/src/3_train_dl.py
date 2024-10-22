import os
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import TextEmbeddingDataset
from model import Net

DEVICE = torch.device("cuda")
Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"

num_epochs = 35
start_list = [0.0, 0.2, 0.4, 0.6, 0.8]
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for sub in os.listdir(Processed_PATH):
    path = os.path.join(Processed_PATH, sub)
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    os.makedirs(result_path, exist_ok=True)

    train_dataset = TextEmbeddingDataset(mode="train", data_path=path)

    for ratio in ratio_list:
        for start in start_list:
            # 指定当前数据的比例以及起始位置
            train_dataset.setState(ratio, start)
            # 获取数据集中向量的形状，用来初始化模型
            DIM = train_dataset.get_dim()
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=64, shuffle=True, drop_last=True
            )

            model = Net(DIM).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=2e-3)  # 初始学习率
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
            model.train()

            for epoch in range(num_epochs):
                running_loss = 0.0
                with tqdm(train_loader, unit="batch") as tepoch:
                    for data in tepoch:
                        inputs, labels = data
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    print(
                        f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}"
                    )
                    scheduler.step()
            # 保存设定下的模型
            torch.save(
                model.state_dict(),
                os.path.join(result_path, f"Net_{int(ratio*10)}_{int(start*10)}.pth"),
            )

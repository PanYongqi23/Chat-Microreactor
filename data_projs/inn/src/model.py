import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x + self.relu(self.bn(self.fc1(x)))
        return self.relu(self.fc2(x))


class Net(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.block1 = ResidualBlock(in_chans, in_chans // 2)
        self.drop1 = nn.Dropout(0.4)
        self.block2 = ResidualBlock(in_chans // 2, in_chans // 4)
        self.drop2 = nn.Dropout(0.4)
        self.block3 = ResidualBlock(in_chans // 4, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.drop1(x)
        x = self.block2(x)
        x = self.drop2(x)
        x = self.block3(x)
        return self.fc(x).squeeze()

    @torch.no_grad()
    def sample_feature_map(self, x):
        x = self.block1(x)
        x = self.drop1(x)
        x = self.block2(x)
        x = self.drop2(x)
        x = self.block3(x)
        return {"feature_map": x, "logits": self.fc(x).squeeze()}

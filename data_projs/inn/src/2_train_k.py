import os
import torch
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"
POSITIVE = 2  # 正类的吸引力
NEGATIVE = -1  # 负类对k的吸引力


# load embedding and label
def load_data(path):
    data = torch.load(path)
    true_mask = data["labels"] == True
    false_mask = data["labels"] == False
    labels = torch.rand(data["labels"].shape)
    labels[true_mask] = POSITIVE
    labels[false_mask] = NEGATIVE
    return data["embeddings"], labels.float()


# compute mean cos similarity on all data, used to evaluate
def compute_cos_similarity(features, k):
    N = len(features)
    point_mul = torch.sum(features * k, dim=1)
    length_k = torch.norm(k)
    length_features = torch.sqrt(torch.sum(features**2, dim=1))

    return torch.sum(point_mul / (length_k * length_features)) / N


# compute gradient by hand to update k
def compute_grad(features, labels, k):
    N = len(features)
    DIM = features.shape[1]
    length_k = torch.norm(k)
    length_features = torch.sqrt(torch.sum(features**2, dim=1))
    k_length_mul_features_length = length_k * length_features

    result = features * (k_length_mul_features_length**2).view(N, 1) - torch.matmul(
        (torch.sum(features * k, dim=1) * length_features**2).view(N, 1), k.view(1, DIM)
    )
    result /= (k_length_mul_features_length**3).view(N, 1)
    result = torch.matmul(labels, result)

    return result.view(1, DIM) / N


# 有一个问题在于不知道在这里要不要区分训练集和测试集
for sub in os.listdir(Processed_PATH):
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    os.makedirs(result_path, exist_ok=True)
    path = os.path.join(Processed_PATH, sub)

    # load data
    features, labels = load_data(path)
    features = features.to(DEVICE)
    labels = labels.to(DEVICE)
    k = torch.rand((1, features.shape[1]), dtype=torch.float32, device=features.device)

    epoch = 250
    lr = 1e-1
    cos_list = []
    with tqdm(range(epoch), unit="epoch") as tepoch:
        for i in tepoch:
            grad = compute_grad(features, labels, k)
            cos = compute_cos_similarity(features, k)
            cos_list.append(cos.item())
            tepoch.set_postfix(cos_similarity=cos.item())
            k += lr * grad
            # normalize k to help converge stably
            k /= torch.norm(k)

    # record average cos similarity during training
    cos_list = np.array(cos_list)
    torch.save(k, os.path.join(result_path, "k.pth"))
    np.save(os.path.join(result_path, "cos_curve.npy"), cos_list)

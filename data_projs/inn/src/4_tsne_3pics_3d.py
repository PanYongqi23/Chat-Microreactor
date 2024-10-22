import os
import torch
import numpy as np
from model import Net
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加3D图支持

DEVICE = torch.device("cuda")
Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"
Plot_PATH = r".\data\plot"

# 修改plot函数，支持绘制3D图像
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(coord, labels, title, save_path):
    classes = ["not useful", "useful"]
    color_dict = {0: "#B0B0B0", 1: "#8887cb"}  # 0为灰色，1为#47af79
    
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 负类，灰色，更透明
    ax.scatter(
        coord[labels == 0, 0],
        coord[labels == 0, 1],
        coord[labels == 0, 2],
        c=color_dict[0],
        label=classes[0],
        s=3,
        alpha=0.2  # 更高的透明度
    )
    
    # 正类，#47af79，较少的透明度
    ax.scatter(
        coord[labels == 1, 0],
        coord[labels == 1, 1],
        coord[labels == 1, 2],
        c=color_dict[1],
        label=classes[1],
        s=3,
        alpha=0.5  # 较低的透明度
    )
    
    ax.set_facecolor('white')
    # 设置网格面板颜色为白色
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))  # x 面板，白色
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))  # y 面板，白色
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))  # z 面板，白色
    
    ax.legend(loc="upper right", fontsize=18, markerscale=3)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("t-SNE Feature 1", fontsize=22)
    ax.set_ylabel("t-SNE Feature 2", fontsize=22)
    ax.set_zlabel("t-SNE Feature 3", fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 不显示坐标轴的数字，但保持网格
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, linestyle="--", alpha=0.1)  # 减少网格的透明度使其更稀疏

    plt.savefig(save_path)


def compute_cos_similarity(features, k):
    point_mul = torch.sum(features * k, dim=1)
    length_k = torch.norm(k)
    length_features = torch.sqrt(torch.sum(features**2, dim=1))

    return point_mul / (length_k * length_features)


for sub in os.listdir(Processed_PATH):
    path = os.path.join(Processed_PATH, sub)
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    plot_path = os.path.join(Plot_PATH, sub.split("_data")[0])
    os.makedirs(plot_path, exist_ok=True)

    data = torch.load(path)
    labels = np.array(data["labels"].int())
    features = np.array(data["embeddings"])

    # 计算数据的降维坐标，改为3维
    tsne = TSNE(n_components=3, random_state=97)
    coord = tsne.fit_transform(features)
    plot_3d(coord, labels, 'Original Data', os.path.join(plot_path, "original_tsne_3d.jpg"))

    k_path = os.path.join(result_path, "k.pth")
    k = torch.load(k_path)
    cos_similarity = compute_cos_similarity(
        torch.from_numpy(features).to(torch.device("cuda")), k
    )
    cos_similarity = np.array(cos_similarity.cpu())
    k_plot_data = {"labels": labels, "cos_similarity": cos_similarity}
    torch.save(k_plot_data, os.path.join(result_path, "k_plot_data.pth"))

    THRESH = 0.6
    predict = (cos_similarity > THRESH).astype(np.int32)
    plot_3d(coord, predict, 'k Predict', os.path.join(plot_path, "K_predict_tsne_3d.jpg"))

    #ratio = 0.9
    ratio = 0.1
    start = 0.0
    model_path = os.path.join(result_path, f"Net_{int(ratio*10)}_{int(start*10)}.pth")
    DIM = data["embeddings"].shape[1]
    model = Net(DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 64
    marker = 0
    predict = None
    feature_map = None
    while True:
        if marker + batch_size > data["embeddings"].shape[0]:
            input = data["embeddings"][marker:].to(DEVICE)
            output = model.sample_feature_map(input)
            predict = torch.hstack([predict, torch.sigmoid(output["logits"]) > 0.5])
            feature_map = torch.vstack([feature_map, output["feature_map"]])
            break

        input = data["embeddings"][marker : marker + batch_size].to(DEVICE)
        output = model.sample_feature_map(input)
        if marker == 0:
            predict = torch.sigmoid(output["logits"]) > 0.5
            feature_map = output["feature_map"]
        else:
            predict = torch.hstack([predict, torch.sigmoid(model(input)) > 0.5])
            feature_map = torch.vstack([feature_map, output["feature_map"]])

        marker += batch_size

    predict = np.array(predict.cpu()).astype(np.int32)
    plot_3d(coord, predict, 'NN Predict', os.path.join(plot_path, "AI_predict_tsne_3d.jpg"))

    tsne = TSNE(n_components=3, random_state=97)
    coord = tsne.fit_transform(np.array(feature_map.cpu()))
    plot_3d(coord, predict, '3D NN Feature Map', os.path.join(plot_path, "AI_feature_map_tsne_3d.jpg"))

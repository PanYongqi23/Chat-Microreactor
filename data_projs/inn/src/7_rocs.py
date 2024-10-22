from sklearn.metrics import roc_curve, auc
import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import TextEmbeddingDataset
from model import Net

Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"
Plot_PATH = r".\data\plot"

# 这里计算的不是平均值，而是每一个数据与k的相似度
# 输出长度为n，对应每一条数据的余弦相似度

DEVICE = torch.device("cuda")


def compute_cos_similarity(features, k):
    point_mul = torch.sum(features * k, dim=1)
    length_k = torch.norm(k)
    length_features = torch.sqrt(torch.sum(features**2, dim=1))

    return point_mul / (length_k * length_features)
    
# 定义颜色和线条风格
k_color = '#5094d5'
nn_color = '#8887cb'
line_style = '--'


# "k-Value Method" 的ROC曲线
for sub in os.listdir(Processed_PATH):
    
    plt.figure(figsize=(10, 8))

    path = os.path.join(Processed_PATH, sub)
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    plot_path = os.path.join(Plot_PATH, sub.split("_data")[0])
    distri_path = os.path.join(plot_path, "ai_distribution")
    os.makedirs(plot_path, exist_ok=True)

    data = torch.load(path)
    labels = np.array(data["labels"].int())
    features = np.array(data["embeddings"])

    k_path = os.path.join(result_path, "k.pth")
    k = torch.load(k_path)
    cos_similarity = compute_cos_similarity(
        torch.from_numpy(features).to(torch.device("cuda")), k
    )
    cos_similarity = np.array(cos_similarity.cpu())

    fpr_k, tpr_k, thresholds_k = roc_curve(labels, cos_similarity)
    roc_auc_k = auc(fpr_k, tpr_k)

    plt.plot(fpr_k, tpr_k, color=k_color, lw=2, label=f'k-Value Method (AUC = {roc_auc_k:.2f})')

    # "NN Method" 的ROC曲线
    
    ratio, start = 0.9, 0.0
    
    test_dataset = TextEmbeddingDataset(mode="test", data_path=path)
    train_dataset = TextEmbeddingDataset(mode="train", data_path=path)
    
    train_dataset.setState(ratio, start)
    test_dataset.setState(ratio, start)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    DIM = train_dataset.get_dim()
    model = Net(DIM).to(DEVICE)
    model.eval()
    model.load_state_dict(
        torch.load(
            os.path.join(
                result_path, f"Net_{int(ratio*10)}_{int(start*10)}.pth"
            )
        )
    )
            
    true_labels = []
    probabilities_list = []

    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs)

        true_labels.append(labels.int())
        probabilities_list.append(probabilities)

    true_labels = torch.hstack(true_labels).detach().cpu()
    probabilities_list = torch.hstack(probabilities_list).detach().cpu()

    fpr_nn, tpr_nn, thresholds_nn = roc_curve(true_labels, probabilities_list)
    roc_auc_nn = auc(fpr_nn, tpr_nn)

    plt.plot(fpr_nn, tpr_nn, color=nn_color, lw=2, label=f'NN Method (AUC = {roc_auc_nn:.2f})')

    # 公共部分：绘制对角线和设置图例
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=line_style)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.title('Receiver Operating Characteristic', fontsize=22)
    plt.legend(loc='lower right', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # 绘制y = 0.95的虚线
    plt.plot([0, 1], [0.95, 0.95], color='#D15354', lw=2, linestyle='--', label='y = 0.95')
    plt.text(-0.081, 0.94, 'TPR = 0.95', color='#D15354', fontsize=18, verticalalignment='top', horizontalalignment='left')


    # 找出虚线与k-Value Method曲线的交点
    # 由于fpr_k和tpr_k是按照fpr排序的，我们可以使用numpy的搜索排序功能
    intersection_k = np.interp(0.95, tpr_k, fpr_k)

    # 在x轴上标出交点的x值
    plt.plot([intersection_k, intersection_k], [0, 0.95], color=k_color, linestyle=':', label=f'k-Value Method intersection (x = {intersection_k:.2f})')
    plt.text(intersection_k, 0.2, f'FPR = {intersection_k:.2f}', color=k_color, fontsize=18, verticalalignment='top', horizontalalignment='center')

    # 找出虚线与NN Method曲线的交点
    intersection_nn = np.interp(0.95, tpr_nn, fpr_nn)

    # 在x轴上标出交点的x值
    plt.plot([intersection_nn, intersection_nn], [0, 0.95], color=nn_color, linestyle=':', label=f'NN Method intersection (x = {intersection_nn:.2f})')
    plt.text(intersection_nn, 0.2, f'FPR = {intersection_nn:.2f}', color=nn_color, fontsize=22, verticalalignment='top', horizontalalignment='center')

    # 保存最终图像
    combined_roc_path = os.path.join(plot_path, "combined_ROC_curve.jpg")
    plt.grid(False) #, linestyle="--", alpha=0.5
    plt.savefig(combined_roc_path) 
    plt.close()

    # 绘制阈值对TPR的曲线
    plt.figure(figsize=(14, 12))

    # 确保cos_similarity是在CPU上的NumPy数组
    #cos_similarity = np.array(cos_similarity)

    # "k-Value Method" 的阈值对TPR曲线
    plt.plot(thresholds_k, tpr_k, color=k_color, lw=2, label=f'k-Value Method TPR vs. Threshold')

    # "NN Method" 的阈值对TPR曲线
    plt.plot(thresholds_nn, tpr_nn, color=nn_color, lw=2, label=f'NN Method TPR vs. Threshold')

    # 设置坐标轴范围，只展示0-1
    plt.xlim(0, 0.0001)
    
    # 设置坐标轴为对数坐标
    #plt.xscale('log')

    # 公共部分：设置图例和坐标轴标签
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('True Positive Rate vs. Threshold', fontsize=16)
    plt.legend(loc='lower left', fontsize=14)

    # 保存最终图像
    threshold_tpr_path = os.path.join(plot_path, "threshold_TPR_curve.jpg")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(threshold_tpr_path)
    plt.close()
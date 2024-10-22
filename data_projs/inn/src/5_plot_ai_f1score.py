import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import TextEmbeddingDataset
from model import Net
#from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda")
Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"
Plot_PATH = r".\data\plot"

threshold = 0.00002
# threshold = 0.5
start_list = [0.0, 0.2, 0.4, 0.6, 0.8]
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 用于绘制ai模型的分布
def plot_ai_distribution(true_labels, probabilities, save_path):
    data1 = np.array(probabilities[true_labels == 1])
    data2 = np.array(probabilities[true_labels == 0])

    plt.figure(figsize=(10, 8))
    plt.hist(
        [data1, data2],
        bins=40,
        color=["#5094d5", "#8887CB"],
        alpha=0.7,
        label=["useful", "not useful"],
        edgecolor="black",
        linewidth=1,
    )

    plt.grid(axis="y", alpha=0.75)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.xlabel("Value", fontsize=22, color="darkred")
    plt.ylabel("Frequency", fontsize=22, color="darkred")

    plt.title("Probability Histogram Comparison", fontsize=22, color="darkred")

    mean1, median1 = np.mean(data1), np.median(data1)
    mean2, median2 = np.mean(data2), np.median(data2)

    plt.text(
        0.15,
        0.95,
        f"positive:\nMean: {mean1:.2f}\nMedian: {median1:.2f}",
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#5094d5", alpha=0.5),
    )

    plt.text(
        0.15,
        0.75,
        f"negative:\nMean: {mean2:.2f}\nMedian: {median2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#bfc7e5", alpha=0.5),
    )

    plt.legend(loc="upper right", fontsize=22)

    plt.savefig(save_path, dpi=300)
    plt.close()


for sub in os.listdir(Processed_PATH):
    path = os.path.join(Processed_PATH, sub)
    print(path)
    plot_path = os.path.join(Plot_PATH, sub.split("_data")[0])
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    distri_path = os.path.join(plot_path, "ai_distribution")
    os.makedirs(distri_path, exist_ok=True)

    test_dataset = TextEmbeddingDataset(mode="test", data_path=path)
    train_dataset = TextEmbeddingDataset(mode="train", data_path=path)

    train_f1_mean_list = []
    train_f1_std_list = []
    test_f1_mean_list = []
    test_f1_std_list = []

    for ratio in ratio_list:
        train_f1_list = []
        test_f1_list = []
        for start in start_list:
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
            pred_labels = []
            probabilities_list = []

            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities >= threshold).int()

                true_labels.append(labels.int())
                pred_labels.append(predicted.int())
                probabilities_list.append(probabilities)

            true_labels = torch.hstack(true_labels).detach().cpu()
            pred_labels = torch.hstack(pred_labels).detach().cpu()

            # 使用 sklearn 计算 test f1-score
            test_f1 = f1_score(true_labels, pred_labels, average='binary')
            test_f1_list.append(test_f1)

            tmp_path = os.path.join(distri_path, f"train_ratio_{ratio}")
            os.makedirs(tmp_path, exist_ok=True)
            plot_ai_distribution(
                true_labels, torch.hstack(probabilities_list).detach().cpu(),
                os.path.join(tmp_path, f"start_{start}.jpg")
            )

            true_labels = []
            pred_labels = []

            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities >= threshold).int()

                true_labels.append(labels.int())
                pred_labels.append(predicted.int())

            # 使用 sklearn 计算 train f1-score
            true_labels = torch.hstack(true_labels).detach().cpu()
            pred_labels = torch.hstack(pred_labels).detach().cpu()
            train_f1 = f1_score(true_labels, pred_labels, average='binary')
            train_f1_list.append(train_f1)

        train_f1_mean_list.append(np.mean(train_f1_list))
        train_f1_std_list.append(np.std(train_f1_list))
        test_f1_mean_list.append(np.mean(test_f1_list))
        print(np.mean(test_f1_list))
        test_f1_std_list.append(np.std(test_f1_list))

    plt.figure(figsize=(10, 8))  # 可以调整图表大小
    plt.errorbar(
        ratio_list,
        train_f1_mean_list,
        yerr=train_f1_std_list,
        fmt="-o",
        label="Train F1",
        color="#5094d5",
    )
    plt.errorbar(
        ratio_list,
        test_f1_mean_list,
        yerr=test_f1_std_list,
        fmt="-s",
        label="Test F1",
        color="#8887cb",
    )
    plt.ylim(0.5, 1)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    plt.title("F1 Score", fontsize=22)
    plt.xlabel("Train Ratio", fontsize=22)
    plt.ylabel("F1 Score", fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.savefig(os.path.join(plot_path, "f1score_errorbar.jpg"))

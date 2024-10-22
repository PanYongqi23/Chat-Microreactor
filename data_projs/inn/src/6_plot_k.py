import os
import torch
import numpy as np
import matplotlib.pyplot as plt

Processed_PATH = r".\data\processed"
Result_PATH = r".\data\result"
Plot_PATH = r".\data\plot"

for sub in os.listdir(Processed_PATH):
    result_path = os.path.join(Result_PATH, sub.split("_data")[0])
    plot_path = os.path.join(Plot_PATH, sub.split("_data")[0])

    k_plot_data = torch.load(os.path.join(result_path, "k_plot_data.pth"))
    labels = k_plot_data["labels"]
    cos_similarity = k_plot_data["cos_similarity"]

    data1 = np.array(cos_similarity[labels == 1])
    data2 = np.array(cos_similarity[labels == 0])

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

    plt.title("Cos Similarity Histogram Comparison", fontsize=22, color="darkred")

    mean1, median1 = np.mean(data1), np.median(data1)
    mean2, median2 = np.mean(data2), np.median(data2)

    plt.text(
        0.04,
        0.95,
        f"positive:\nMean: {mean1:.2f}\nMedian: {median1:.2f}",
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#5094d5", alpha=0.5),
    )

    plt.text(
        0.04,
        0.75,
        f"negative:\nMean: {mean2:.2f}\nMedian: {median2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#bfc7e5", alpha=0.5),
    )

    plt.legend(loc="upper right", fontsize=22)
    plt.savefig(os.path.join(plot_path, "cos_distribution.jpg"), dpi=300)
    plt.close()

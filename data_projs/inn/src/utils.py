import torch
import torch.utils

class TextEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_path):
        super().__init__()
        self.mode = mode
        self.raw_data = torch.load(data_path)
        self.raw_data["labels"] = self.raw_data["labels"].float()

    def setState(self, ratio, start):
        N = len(self.raw_data["labels"])
        self.features = torch.vstack(
            [
                self.raw_data["embeddings"][int(start * N) :],
                self.raw_data["embeddings"][: int(start * N)],
            ]
        )
        self.labels = torch.hstack(
            [
                self.raw_data["labels"][int(start * N) :],
                self.raw_data["labels"][: int(start * N)],
            ]
        )
        if self.mode == "train":
            self.features = self.features[: int(ratio * N)]
            self.labels = self.labels[: int(ratio * N)]
        elif self.mode == "test":
            self.features = self.features[int(ratio * N) :]
            self.labels = self.labels[int(ratio * N) :]

    def setState(self,ratio,start):
        N=len(self.raw_data['labels'])
        self.features=torch.vstack([self.raw_data['embeddings'][int(start*N):],self.raw_data['embeddings'][:int(start*N)]])
        self.labels=torch.hstack([self.raw_data['labels'][int(start*N):],self.raw_data['labels'][:int(start*N)]])
        if self.mode=='train':
            self.features=self.features[:int(ratio*N)]
            self.labels=self.labels[:int(ratio*N)]
        elif self.mode=='test':
            self.features=self.features[int(ratio*N):]
            self.labels=self.labels[int(ratio*N):]
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    def get_dim(self):
        return self.features.shape[1]

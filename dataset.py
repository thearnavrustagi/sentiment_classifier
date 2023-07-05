import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

class SentimentDataset (Dataset):
    def __init__ (self, data_file="./dataset/dataset.csv"):
        self.dataframe = pd.read_csv(data_file,encoding="latin").dropna().reset_index(drop=True)
        tdf = self.dataframe.copy(deep=True)

        self.text = tdf.pop("text").to_numpy()
        self.sentiment = tdf.pop("sentiment").to_numpy()

        self.tokenizer = Tokenizer.load()

        print(self.text.shape, self.sentiment.shape,len(self.tokenizer.word_to_idx))

    def __len__ (self):
        return self.sentiment.shape[0]

    def __getitem__ (self, idx):
        data = self.dataframe.iloc[idx].tolist()
        x_train = self.tokenizer.to_tensor(data[5])

        y_train = int(data[0]/4)
        return (x_train, y_train)

if __name__ == "__main__":
    sd = SentimentDataset()
    print(len(sd))
    print(sd[0])

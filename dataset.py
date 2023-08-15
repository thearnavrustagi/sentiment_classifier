import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

from constants import DATASET_FNAME, VALIDATION_FNAME
from constants import VALIDATION_DS_ROWS

# https://www.kaggle.com/datasets/kazanova/sentiment140
class SentimentDataset (Dataset):
    def __init__ (self, data_file=DATASET_FNAME):
        self.dataframe = pd.read_csv(data_file,encoding="latin").dropna().reset_index(drop=True)
        #self.dataframe = self.dataframe[ (self.dataframe['category'] != 0) ]
        #self.dataframe = self.dataframe.iloc[0:VALIDATION_DS_ROWS]
        tdf = self.dataframe.copy(deep=True)

        self.text = tdf.pop("text").to_numpy()
        self.sentiment = tdf.pop("target").to_numpy()

        self.tokenizer = Tokenizer.load()

        print(self.text.shape, self.sentiment.shape,len(self.tokenizer.word_to_idx))

    def __len__ (self):
        return self.sentiment.shape[0]

    def __getitem__ (self, idx):
        data = self.dataframe.iloc[idx].tolist()
        x_train = self.tokenizer.to_tensor(self.text[idx])

        y_train = int(self.sentiment[idx]/2)
        return (x_train, y_train)

if __name__ == "__main__":
    sd = SentimentDataset()
    tdf = sd.dataframe.sample(frac=1)
    df = tdf.iloc[0:VALIDATION_DS_ROWS]
    print(df)
    df.to_csv(VALIDATION_FNAME)
    print(len(sd))
    print(sd[0])

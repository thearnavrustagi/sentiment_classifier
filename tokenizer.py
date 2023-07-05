import torch
import pickle
import pandas as pd

from constants import VOCAB_SIZE, UNKNOWN_TOKEN, DATASET_FNAME
from constants import END_TOKEN, PAD_TOKEN, SENTENCE_LEN
from utils import text_preprocess

class Tokenizer (object):
    def __init__ (self, preprocess=text_preprocess, vocabulary_size=VOCAB_SIZE):
        self.preprocess = preprocess
        self.vocabulary_size = int(vocabulary_size)

        self.word_to_idx = {}
        self.idx_to_word = []

    def __call__ (self,data):
        count = []
        self.word_to_idx = {}
        for sent in data:
            try:
                sent = self.preprocess(sent)
                for word in sent:
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = len(count)
                        count.append((word,0))
                    i = self.word_to_idx[word]
                    _,n = count[i]
                    count[i] = (word,n+1)
            except Exception as e:
                print(e)

        self.idx_to_word = sorted(count,key=lambda x : -x[1])[:self.vocabulary_size-2]
        self.idx_to_word = [elem[0] for elem in self.idx_to_word] + [UNKNOWN_TOKEN] + [PAD_TOKEN]

        self.word_to_idx = {}
        for i,w in enumerate(self.idx_to_word):
            self.word_to_idx[w] = i

    def to_tensor (self,sent):
        final = []
        true_len = 0
        ended = True
        for word in self.preprocess(sent):
            if true_len == SENTENCE_LEN: break
            if word != PAD_TOKEN: true_len += 1
            if word == PAD_TOKEN: ended = True
            if word in self.word_to_idx:
                final.append(self.word_to_idx[word])
            else : final.append(self.word_to_idx[UNKNOWN_TOKEN])
        if not ended:
            final[-1] = self.word_to_idx(END_TOKEN)
        return (torch.tensor(final),true_len)

    def to_sentence (self,sent):
        return " ".join([self.idx_to_word[i] for i in list(sent)[1:-1]])

    def save(self,filename="./model/tokenizer.pkl"):
        with open (filename, 'wb') as file:
            pickle.dump(self,file,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load (filename="./model/tokenizer.pkl"):
        with open(filename,"rb") as file:
            return pickle.load(file)

if __name__ == "__main__":
    data = """
    Isn’t it greedy?
    In order to represent the corpus in the most efficient way, BPE goes through every potential option to merge at each iteration by looking at its frequency. So, yes it follows a greedy approach to optimize for the best possible solution.
    Anyways, BPE is one of the most widely used subword-tokenization algorithms and it has a good performance despite being greedy.  """.split('\n')
    tokenizer = Tokenizer (vocabulary_size=50)
    tokenizer(data)
    print(len(tokenizer.word_to_idx))
    print(len(tokenizer.idx_to_word))

    s = "In the most efficient through every"
    print(s)
    t = tokenizer.to_tensor(s)
    print(t)
    print(tokenizer.to_sentence(t))


    print("="*80)
    print("training tokenizer")
    tokenizer = Tokenizer ()
    data = pd.read_csv(DATASET_FNAME,encoding="latin").pop("text").tolist()
    tokenizer(data)
    s = "In the most efficient through every"
    print(s)
    t = tokenizer.to_tensor(s)
    print(t)
    tokenizer.save()

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from dataset import SentimentDataset
from tokenizer import Tokenizer
from model import SentimentModel

from constants import BATCH_SIZE, N_EPOCHS, VOCAB_SIZE, LEARNING_RATE
from constants import WORKERS

def main ():
    sentiment_dataset = SentimentDataset()
    dataloader = DataLoader(sentiment_dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers=WORKERS)
    model = None 
    try:
        model = SentimentModel.load()
        print("loaded model")
    except:
        print("failed to load model")
        model = SentimentModel()
    finally:
        print(model)

    try:
        train(model,dataloader)
    except KeyboardInterrupt as e:
        print(e)
        model.save()

def train (model,dataloader, lr = LEARNING_RATE, n_epochs = N_EPOCHS):
    optimizer = Adam(model.parameters(), lr)
    loss_fn = CrossEntropyLoss()
    model.train()

    for epoch in range(n_epochs):
        n_minibatches = len(dataloader)
        with tqdm(total=(n_minibatches),position=0, leave=True) as prog:
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                loss = 0.
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

                # prog bar
                prog.update(1)
                prog.set_description(f"loss : {loss}")
        print("saving model")
        model.save()

        print(f"epoch {epoch+1} done")


if __name__ == "__main__":
    main()


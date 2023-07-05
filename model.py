import torch
from torch import nn

from constants import SENTENCE_LEN, VOCAB_SIZE

class SentimentModel (nn.Module):

    def __init__ (self,
                  padding_idx : int = VOCAB_SIZE-1,
                  vocabulary_size : int = VOCAB_SIZE,
                  embed_depth : int = 256,
                  hidden_size : int = 256,
                  lstm_layers : int = 2,
                  dropout : float = 0.15,
                  classes : int = 2,
                  sent_len : int = SENTENCE_LEN,
                  attention_heads : int = 16,
                  ):
        super(SentimentModel,self).__init__()
        vocabulary_size = int(vocabulary_size)

        self.vocabulary_size = vocabulary_size
        self.embed_depth = embed_depth
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.classes = classes
        self.attention_heads = attention_heads
        self.sent_len = SENTENCE_LEN
        
        self.embedding = nn.Embedding(vocabulary_size, embed_depth,padding_idx)

        self.self_attention = nn.MultiheadAttention(embed_depth, attention_heads, bias=True)
        self.layer_norm = nn.LayerNorm(embed_depth)
        self.lstm = nn.LSTM(
                input_size = embed_depth,
                hidden_size = hidden_size,
                num_layers = lstm_layers,
                dropout = dropout,
                bias = True,
                bidirectional = True
                )
        self.relu = nn.ReLU()

        # linear layer applied on the hidden state vector from the LSTM
        self.output_projection = nn.Linear (hidden_size*2*self.sent_len, classes, bias = True)

    
    def forward (self, x):
        x, lens = x
        x = self.embedding(x)
        attn_output,attn_output_weights = self.self_attention(query=x,key=x,value=x)
        x = x + attn_output
        x = self.layer_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True,enforce_sorted=False) # (batch * seqlens)
        x,_ = self.lstm(x)
        x,lens = nn.utils.rnn.pad_packed_sequence(x,batch_first=True,total_length=SENTENCE_LEN)
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.relu(x)
        x = self.output_projection(x)

        return x

    def save (self,savepath="./model/sentiment_model.dat"):
        torch.save(self.state_dict(),savepath)

    @staticmethod
    def load (path="./model/sentiment_model.dat"):
        model = SentimentModel()
        model.load_state_dict(torch.load(path))
        model.eval()

        return model

if __name__ == "__main__":
    pass

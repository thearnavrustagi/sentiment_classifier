from bpemb import BPEmb

class Embedder (object):
    def __init__ (self,vocabulary_size=1e5,embed_depth=1024,lang="en"):
        self.__embedder__ = BPEmb(lang=lang, dim=embed_depth)#, vs = vocabulary_size)
        self.embed_depth

    # encodes the data to integers
    def encode (self, x):
        return self.__embedder__.encode_ids(x)

    # requires a tensor of already encoded data
    def embed (self,x):
        return self.__embedder__.vectors[x]

if __name__ == "__main__":
    print("testing embedder")
    embedder = Embedder()
    data = "Hello ! I am nullpointer exception here with an LSTM, which means long short term memory".split()
    print(f"data : {data}")
    x = embedder.encode(data)
    print("encoded : {x}")
    x = embedder.embed(x)
    print("embedded : {x.shape}")


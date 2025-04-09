class CharTokenizer:
    def __init__(self, vocab):
        self.tok2idx = {ch: i + 1 for i, ch in enumerate(sorted(vocab))}
        self.tok2idx["<pad>"] = 0
        self.tok2idx["<sos>"] = len(self.tok2idx)
        self.tok2idx["<eos>"] = len(self.tok2idx)
        self.idx2tok = {i: ch for ch, i in self.tok2idx.items()}

    def encode(self, text, max_len):
        tokens = (
            [self.tok2idx["<sos>"]]
            + [self.tok2idx[ch] for ch in text if ch in self.tok2idx]
            + [self.tok2idx["<eos>"]]
        )
        tokens += [self.tok2idx["<pad>"]] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, indices):
        return "".join(
            self.idx2tok[idx]
            for idx in indices
            if idx in self.idx2tok
            and idx not in [0, self.tok2idx["<sos>"], self.tok2idx["<eos>"]]
        )

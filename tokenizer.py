class CharTokenizer:
    def __init__(self, chars):
        special_tokens = ["<pad>", "<sos>", "<eos>"]
        unique_chars = sorted(set(chars))
        self.char2idx = {
            ch: i + len(special_tokens) for i, ch in enumerate(unique_chars)
        }
        for i, token in enumerate(special_tokens):
            self.char2idx[token] = i

        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.pad_idx = self.char2idx["<pad>"]
        self.sos_idx = self.char2idx["<sos>"]
        self.eos_idx = self.char2idx["<eos>"]

    def encode(self, text):
        return (
            [self.sos_idx]
            + [self.char2idx.get(c, self.pad_idx) for c in text]
            + [self.eos_idx]
        )

    def decode(self, indices):
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, "")
            if char in ("<sos>", "<eos>", "<pad>"):
                continue
            chars.append(char)
        return "".join(chars)

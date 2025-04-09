from collections import defaultdict
from typing import Counter


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.tokens = set()

    def get_vocab(self):
        return self.tok2idx

    def train(self, corpus):
        """
        corpus: list of strings (training sentences)
        """
        # Split each word into characters + </w> to mark end-of-word
        word_freqs = Counter()
        for line in corpus:
            for word in line.strip().split():
                word = " ".join(list(word)) + " </w>"
                word_freqs[word] += 1

        vocab = dict(word_freqs)

        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            return pairs

        def merge_vocab(pair, vocab):
            pattern = re.escape(" ".join(pair))
            replacement = "".join(pair)
            new_vocab = {}
            for word in vocab:
                new_word = re.sub(pattern, replacement, word)
                new_vocab[new_word] = vocab[word]
            return new_vocab

        # Learn BPE merges
        for _ in range(self.vocab_size):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            self.bpe_codes[best] = len(self.bpe_codes)

        # Extract final tokens
        tokens = set()
        for word in vocab:
            tokens.update(word.split())

        tokens = sorted(tokens)
        super().__init__(tokens)

    def apply_bpe(self, word):
        word = list(word) + ["</w>"]
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            mergeable = [
                (pair, self.bpe_codes.get(pair, float("inf"))) for pair in pairs
            ]
            mergeable = [p for p in mergeable if p[1] != float("inf")]
            if not mergeable:
                break
            best_pair = min(mergeable, key=lambda x: x[1])[0]

            # Merge best pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text, max_len):
        words = text.strip().split()
        tokens = [self.tok2idx["<sos>"]]
        for word in words:
            bpe_tokens = self.apply_bpe(word)
            for tok in bpe_tokens:
                if tok in self.tok2idx:
                    tokens.append(self.tok2idx[tok])
        tokens.append(self.tok2idx["<eos>"])
        tokens += [self.tok2idx["<pad>"]] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in self.idx2tok:
                tok = self.idx2tok[idx]
                if tok in ["<pad>", "<sos>", "<eos>"]:
                    continue
                if tok.endswith("</w>"):
                    words.append(tok[:-4])
                else:
                    words.append(tok)
        return "".join(words)

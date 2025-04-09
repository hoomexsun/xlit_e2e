__all__ = ["build_tokenizer"]


from pathlib import Path

import yaml


def build_tokenizer(xs, token_type="char", lang="mni", vocab_size=1000):
    import yaml

    if token_type == "char":
        from .char_tokenizer import CharTokenizer

        tokenizer = CharTokenizer(set("".join(xs)))
    elif token_type == "bpe":
        from .bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(xs)
    else:
        raise ValueError(f"Unsupported token type: {token_type}")

    exp_path = Path(f"exp/{lang}_{token_type}_tokenizer.yaml")
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(exp_path, "w", encoding="utf-8") as f:
        yaml.dump(tokenizer.tok2idx, f, allow_unicode=True)
    print(f"Tokenizer saved to {exp_path}")

    return tokenizer


def load_tokenizer(token_type="char", lang="mni"):
    import yaml

    tokenizer_path = Path(f"exp/{lang}_{token_type}_tokenizer.yaml")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok2idx = yaml.safe_load(f)

    # YAML will keep int keys if they were int, otherwise convert
    if token_type == "char":
        from .char_tokenizer import CharTokenizer

        tokenizer = CharTokenizer(vocab=[])
        tokenizer.tok2idx = tok2idx
        tokenizer.idx2tok = {i: ch for ch, i in tok2idx.items()}
    elif token_type == "bpe":
        from .bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.tok2idx = tok2idx
        tokenizer.idx2tok = {i: ch for ch, i in tok2idx.items()}
    else:
        raise ValueError(f"Unsupported token type: {token_type}")

    print(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

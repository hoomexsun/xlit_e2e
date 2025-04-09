import yaml


def load_data(filepath, params_file):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pairs = [line.strip().split() for line in lines if len(line.strip().split()) == 2]
    xs, ys = zip(*pairs)

    x_vocab = set("".join(xs))
    y_vocab = set("".join(ys))
    max_len = max((max(map(len, xs)), max(map(len, ys)))) + 2
    embed_dim = len(x_vocab) // 2
    hidden_dim = max(len(x_vocab), len(y_vocab))
    batch_size = min(32, len(xs) // 10)

    params = {
        "x_vocab": list(x_vocab),
        "y_vocab": list(y_vocab),
        "max_len": max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
    }

    store_yaml(params_file, params)
    return list(xs), list(ys), params


def store_yaml(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

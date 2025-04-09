from .seq2seq import Seq2Seq
from .rnn import RNNModel
from .cnn import CNNModel
from .rule_based import RBModel

__all__ = ["Seq2Seq", "RNNModel", "CNNModel", "RBModel"]


def get_model_class(xlit_type: str):
    model_map = {
        "seq2seq": Seq2Seq,
        "rnn": RNNModel,
        "cnn": CNNModel,
        "rule": RBModel,
    }

    if xlit_type not in model_map:
        raise ValueError(
            f"Unknown model type '{xlit_type}'. Available: {list(model_map.keys())}"
        )
    return model_map[xlit_type]

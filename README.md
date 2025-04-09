# xlit_e2e

End to End Transliteration

## Directory structure

```perl
xlit_project/
│
├── conf/
│   ├── train.yaml
│   ├── decode.yaml
│   └── tuning/
│       ├── char_cnn.yaml
│       ├── char_dnn.yaml
│       ├── char_rnn.yaml
│       ├── char_seq2seq.yaml
│       ├── bpe_seq2seq.yaml
│       ├── custom_cnn.yaml
│       └── ... (other configurations)
│
├── data/
│   ├── train.txt
│   ├── test.txt
│   └── config.json        # Contains vocab and max_len
│
├── exp/
│   ├── char_seq2seq/      # Format: <tokenizer>_<model>
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── tensorboard/
│   │   └── images/
│   ├── bpe_rnn/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── tensorboard/
│   │   └── images/
│   └── ... (other experiments)
│
├── models/
│   ├── __init__.py
│   ├── base.py               # Base class for models
│   ├── cnn.py
│   ├── rnn.py
│   ├── dnn.py
│   ├── seq2seq.py
│   └── attention.py          # Shared attention modules
│
├── tokenizers/
│   ├── __init__.py
│   ├── base.py               # BaseTokenizer class
│   ├── char_tokenizer.py
│   ├── bpe_tokenizer.py
│   └── custom_tokenizer.py
│
├── dataset/
│   ├── __init__.py
│   ├── dataset.py
│   └── utils.py              # load_data and helper functions
│
├── checkpoint/
│   ├── save.py
│   └── load.py
│
├── utils/
│   ├── train_utils.py        # set_seed, evaluate, inference, etc.
│   ├── logger.py             # setup_logging
│   └── plot.py               # plot_loss and visualizations
│
├── gui/
│   └── app.py                # GUI application (e.g., Tkinter or PyQt5)
│
├── notebooks/
│   └── demo.ipynb
│
├── main.py                   # Entry point for training & inference
└── README.md
```

| Folder             | Description                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `conf/`            | Contains configuration files for training, decoding, and model-specific tuning.                                              |
| `data/`            | Holds the dataset files and a config.json with metadata like vocabulary and maximum sequence length.                         |
| `exp/`             | Stores experiment artifacts, organized into subdirectories each containing                                                   |
| `exp/checkpoints/` | Model checkpoints saved during training.                                                                                     |
| `exp/logs/`        | Training logs.                                                                                                               |
| `exp/tensorboard/` | TensorBoard log files for visualization.                                                                                     |
| `exp/images/`      | Plots and visualizations related to the experiment.                                                                          |
| `models/`          | Includes the base model class and specific model implementations.                                                            |
| `tokenizers/`      | Contains tokenizer classes, including a base tokenizer and specific implementations like character-level and BPE tokenizers. |
| `dataset/`         | Manages dataset-related scripts and utilities.                                                                               |
| `checkpoint/`      | Handles saving and loading of model checkpoints.                                                                             |
| `utils/`           | Provides utility functions for training, logging, and plotting.                                                              |
| `gui/`             | Optional GUI application for user interaction.                                                                               |
| `notebooks/`       | Jupyter notebooks for demonstrations and experiments.                                                                        |
| `main.py`          | The main script for training and inference operations.                                                                       |

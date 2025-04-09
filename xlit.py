import torch

from models import get_model_class


class XlitE2E:

    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device

        model_class = get_model_class(config["xlit"])
        self.model = model_class(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["xlit_conf"]["optim_conf"]["lr"]
        )
        self.criterion = torch.nn.CrossEntropyLoss()

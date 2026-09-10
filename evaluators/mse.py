import torch

from utils.misc import all_gather
from itertools import chain
from torchmetrics.functional import mean_squared_error

class MSEEvaluator:
    def __init__(
        self,
        std=None,
    ) -> None:
        self.preds = []
        self.tgts = []
        self.std = std

    def update(self, outputs, targets):
        self.preds.extend(outputs.detach().cpu().numpy().flatten())
        self.tgts.extend(targets.detach().cpu().numpy().flatten())

    def compute(
        self,
    ):
        all_preds = all_gather(self.preds)
        all_tgts = all_gather(self.tgts)
        all_preds = list(chain.from_iterable(all_preds))
        all_tgts = list(chain.from_iterable(all_tgts))
        all_preds = torch.tensor(all_preds)
        all_tgts = torch.tensor(all_tgts)

        if len(all_preds) == 0:
            return {
                "mean_squared_error": torch.tensor(torch.nan),
            }

        # all_preds, all_tgts = torch.concat(all_preds, dim=0), torch.concat(
        #     all_tgts, dim=0
        # )

        out = {
            "mean_squared_error": mean_squared_error(all_preds, all_tgts),
        }

        if self.std:
            out["error_in_orig_scale"] = (
                torch.sqrt(out["mean_squared_error"]) * self.std
            )

        return out

from typing import Any, Dict, Optional
import torch
from torch import Tensor


class EpochFlag(Enum):
    """Flags indicating the position within an epoch."""
    START = "start"
    CONTINUE = "continue"
    END = "end"
    START_AND_END = "start_and_end"


@dataclass
class EpochContext:
    """Context information about the current epoch."""
    i_epoch: int = 0
    epoch_flag: Optional[EpochFlag] = None
    i: int = 0  # index within epoch
    epoch_len: int = 0  # total length of epoch

    @property
    def epoch_end(self) -> bool:
        """Check if this is the end of an epoch."""
        return self.epoch_flag in (EpochFlag.END, EpochFlag.START_AND_END)

    @property
    def epoch_start(self) -> bool:
        """Check if this is the start of an epoch."""
        return self.epoch_flag in (EpochFlag.START, EpochFlag.START_AND_END)


def training_data_generator(trainloader, pbar):
    """Generator that yields training data with epoch boundary flags."""
    i_epoch = 0
    pbar_iter = pbar.__iter__()
    while True:
        epoch_len = len(trainloader)
        for i, data in enumerate(trainloader):
            try:
                step = next(pbar_iter)
            except StopIteration:
                return

            if epoch_len == 1:
                epoch_flag = EpochFlag.START_AND_END
            elif i == 0:
                epoch_flag = EpochFlag.START
            elif i == epoch_len - 1:
                epoch_flag = EpochFlag.END
            else:
                epoch_flag = EpochFlag.CONTINUE

            epoch_ctx = EpochContext(
                i_epoch=i_epoch,
                epoch_flag=epoch_flag,
                i=i,
                epoch_len=epoch_len
            )

            yield step, data, epoch_ctx

        i_epoch += 1

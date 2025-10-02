from typing import Dict, Tuple, NamedTuple
import torch



class ForwardStep(NamedTuple):
    loss_total: torch.Tensor
    loss_by_level: TensorByLevel
    logits_by_level: TensorByLevel
    labels_by_level: TensorByLevel
    batch_size: int
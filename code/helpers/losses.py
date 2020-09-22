import numpy as np
import torch
from typing import Sequence, Union
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics import Metric, Loss, MeanSquaredError, RootMeanSquaredError
import torch.nn as nn
import torch.nn.functional as F
import math


class CombinedCustomLoss():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
        self.wse = WeightedSignedErrorLoss(alpha=self.alpha)
        self.smape = SymmetricMeanAbsolutePercentageErrorLoss()
        
    def __call__(self, y_pred, y):
        y_true = y.view_as(y_pred)
        
        return self.wse(y_pred, y_true) + self.smape(y_pred, y_true)

    

class WeightedSignedErrorLoss():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, y_pred, y):
        diff = y_pred - y.view_as(y_pred)
        
        squared_errors = torch.pow(diff, 2)
        
        signs = torch.sign(diff)
        signs[signs == 0] = 1
        
        return torch.mean((self.alpha**((1+signs) / 2)) * squared_errors)
    
    
    
class WeightedSignedSymmetricMeanAbsolutePercentageErrorLoss():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, y_pred, y):
        y_true = y.view_as(y_pred)
        diff = y_pred - y_true
        
        squared_errors = torch.pow(diff, 2)
        
        signs = torch.sign(diff)
        signs[signs == 0] = 1
        
        return torch.mean((self.alpha**((1+signs) / 2)) *(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred))))
    
    
    
class SymmetricMeanAbsolutePercentageErrorLoss():
    def __call__(self, y_pred, y):
        y_true = y.view_as(y_pred)
        
        return torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))
    
    def __repr__(self):
        return "SMAPE"
    
    def __str__(self):
        return "SMAPE"
        
    
class WeightedSignedError(Metric):
    """
    Calculates the weighted signed error.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha 
        super(WeightedSignedError, self).__init__()

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = 0.0
        self._num_examples = 0
        super(WeightedSignedError, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        diff = y_pred - y.view_as(y_pred)
        
        squared_errors = torch.pow(diff, 2)
        
        signs = torch.sign(diff)
        signs[signs == 0] = 1
        
        self._sum_of_errors += torch.sum((self.alpha**((1+signs) / 2)) * squared_errors).item()
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("WeightedSignedError must have at least one example before it can be computed.")
        return self._sum_of_errors / self._num_examples
    
    

class RootWeightedSignedError(WeightedSignedError):
    """
    Calculates the root weighted signed error.

    - ``update`` must receive output of the form (y_pred, y) or `{'y_pred': y_pred, 'y': y}`.
    """
    def __init__(self, alpha=0.2):
        super(RootWeightedSignedError, self).__init__(alpha=alpha)

    def compute(self) -> Union[torch.Tensor, float]:
        wse = super(RootWeightedSignedError, self).compute()
        return math.sqrt(wse)    

    
    
class SymmetricMeanAbsolutePercentageError(Metric):
    """
    Calculates symmetric mean absolute percentage error.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = 0.0
        self._num_examples = 0
        super(SymmetricMeanAbsolutePercentageError, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        y_true = y.view_as(y_pred)
        
        errors = torch.sum(torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred)))
        
        self._sum_of_errors += errors.item()
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("SymmetricMeanAbsolutePercentageError must have at least one example before it can be computed.")
        return self._sum_of_errors / self._num_examples
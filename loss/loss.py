import torch
from torch.nn import Module


class HindsightLoss(Module):
    def forward(self, prediction, target):
        subsets, number_of_predictions = prediction.size()
        loss = torch.nn.BCELoss()
        losses = torch.empty(number_of_predictions)
        for i in range(number_of_predictions):
            losses[i] = loss(prediction[:, i], target)
        return torch.min(losses)
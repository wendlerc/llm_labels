import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputMSE(nn.Module):
    def __init__(self, reduction='sum'):# for some reason sum works much better than mean
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_emb, label_emb, label_logits, label):
        if self.reduction == 'mean':
            return 64 * F.mse_loss(pred_emb, label_emb, reduction=self.reduction)
        return F.mse_loss(pred_emb, label_emb, reduction=self.reduction)


class OutputCE(nn.Module):
    def __init__(self, class_embeddings, temperature=0.075):
        super().__init__()
        self.temperature = temperature
        self.class_embeddings_tensor = class_embeddings

    def forward(self, pred_emb, label_emb, label_logits, label):
        logits = pred_emb @ self.class_embeddings_tensor.T
        if self.temperature == 0:
            target = label
        else:
            target = F.softmax((1/self.temperature)*label_logits, dim=1)
        loss = F.cross_entropy(logits, target) # this requires torch=1.11.0
        return loss



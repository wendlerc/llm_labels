import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputMSE(nn.Module):
    def __init__(self, reduction='sum'):# for some reason sum works much better than mean
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_emb, label_emb):
        if self.reduction == 'mean':
            return 64 * F.mse_loss(pred_emb, label_emb, reduction=self.reduction)
        return F.mse_loss(pred_emb, label_emb, reduction=self.reduction)


class OutputCE(nn.Module):
    def __init__(self, class_embeddings):
        super().__init__()
        self.class_embeddings_tensor = class_embeddings

    def forward(self, pred_emb, label_emb):
        logits = pred_emb @ self.class_embeddings_tensor.T
        y = torch.argmax(label_emb @ self.class_embeddings_tensor.T, dim=1)
        loss = F.cross_entropy(logits, y)
        return loss



import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputCosLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.sim = nn.CosineSimilarity(dim=1)
        self.reduction = reduction

    def forward(self, pred_emb, label_emb, label_logits, label, class_embeddings_tensor=None):
        losses = 1-self.sim(pred_emb, label_emb)
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")


class OutputMSE(nn.Module):
    def __init__(self, reduction='sum'):# for some reason sum works much better than mean
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_emb, label_emb, label_logits, label, class_embeddings_tensor=None):
        if self.reduction == 'mean':
            return 64 * F.mse_loss(pred_emb, label_emb, reduction=self.reduction)
        return F.mse_loss(pred_emb, label_emb, reduction=self.reduction)


class OutputCE(nn.Module):
    def __init__(self, class_embeddings, temperature=0.075):
        super().__init__()
        self.temperature = temperature
        self.class_embeddings_tensor = class_embeddings

    def forward(self, pred_emb, label_emb, label_logits, label, class_embeddings_tensor=None):
        if class_embeddings_tensor is None:
            class_embeddings_tensor = self.class_embeddings_tensor
        logits = pred_emb @ class_embeddings_tensor.T

        if self.temperature == 0:
            target = label
        else:
            target = F.softmax((1/self.temperature)*label_logits, dim=1)
        loss = F.cross_entropy(logits, target) # this requires torch=1.11.0
        return loss



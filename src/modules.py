import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, confusion_matrix
import wandb


def create_model(n_outputs):
    model = torchvision.models.resnet18(pretrained=False, num_classes=n_outputs)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


def my_create_model(n_outputs):
    model = torchvision.models.resnet18(pretrained=False, num_classes=n_outputs)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.avgpool = nn.Identity()
    model.fc = nn.Linear(8192, n_outputs)
    return model


class BaseModule(LightningModule):
    def __init__(self, optimizer='sgd',
                 lr=0.05,
                 weight_decay=5e-4,
                 momentum=0.9,
                 scheduler='one_cycle',
                 three_phase=True,
                 pct_start=0.1,
                 max_lr=0.1,
                 batch_size=256,
                 n_train=45000):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'normalize'])

    def evaluate(self, batch, stage=None):
        x, (emb_y, logits_y, y) = batch
        preds = self(x)  # expects forward to compute class logits
        loss = F.cross_entropy(preds, y)
        acc = accuracy(torch.argmax(preds, dim=1), y)
        acc2 = accuracy(preds, y, top_k=2)
        acc3 = accuracy(preds, y, top_k=3)
        acc4 = accuracy(preds, y, top_k=4)
        acc5 = accuracy(preds, y, top_k=5)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_ce_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc@2", acc2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc@3", acc3, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc@4", acc4, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc@5", acc5, prog_bar=True, on_step=False, on_epoch=True)


        return {'ce_loss': loss, 'acc': acc, 'preds':preds, 'y':y}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs):
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_y = torch.cat([x['y'] for x in outputs], dim=0)
        self.trainer.logger.experiment.log(
            {'confusion_matrix': wandb.Image(confusion_matrix(all_preds, all_y, all_preds.shape[1], normalize='true'))})

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def test_epoch_end(self, outputs):
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_y = torch.cat([x['y'] for x in outputs], dim=0)
        self.trainer.logger.experiment.log(
            {'confusion_matrix': wandb.Image(confusion_matrix(all_preds, all_y, all_preds.shape[1], normalize='true'))})

    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optimizer}")

        if self.hparams.scheduler == 'one_cycle':
            steps_per_epoch = self.hparams.n_train // self.hparams.batch_size
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    self.hparams.max_lr,
                    pct_start=self.hparams.pct_start,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=steps_per_epoch,
                    three_phase=self.hparams.three_phase,
                ),
                "interval": "step",
            }
        elif self.hparams.scheduler is None:
            scheduler_dict = None
        else:
            raise ValueError(f"Unknown scheduler {self.hparams.scheduler}")

        if scheduler_dict is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class OurLitResnet(BaseModule):
    def __init__(self, class_embeddings_tensor, loss, normalize=False, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'normalize'])
        self.model = my_create_model(class_embeddings_tensor.shape[1])
        self.class_embeddings_tensor = class_embeddings_tensor
        self.loss = loss
        self.normalize = normalize

    def forward(self, x):
        embedding = self.model(x)
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        logits = embedding @ self.class_embeddings_tensor.T
        return logits

    def training_step(self, batch, batch_idx):
        x, (emb_y, logits_y, y) = batch
        assert self.model.training
        embedding = self.model(x)
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        loss = self.loss(embedding, emb_y, logits_y, y)
        self.log("train_loss", loss)
        return loss


class LitResnet(BaseModule):
    def __init__(self, n_classes=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, (emb_y, logits_y, y) = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss






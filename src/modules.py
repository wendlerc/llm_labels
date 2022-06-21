import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, confusion_matrix
import wandb
import numpy as np

# this maps from sorted indices to the unsorted ones where the superclasses are in chunks of 5
cifar100_idcs = np.asarray([4, 31, 55, 72, 95, 1, 33, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 29, 61, 0, 51, 53, 57, 83, 22, 25, 40, 86, 87, 5, 20, 26, 84, 94, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 12, 17, 38, 68, 76, 23, 34, 49, 60, 71, 15, 19, 21, 32, 39, 35, 63, 64, 66, 75, 27, 45, 77, 79, 99, 2, 11, 36, 46, 98, 28, 30, 44, 78, 93, 37, 50, 65, 74, 80, 47, 52, 56, 59, 96, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89])


def create_projection_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model


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
                 steps_per_epoch=None):
        super().__init__()
        self.stage = None
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'normalize', 'test_classes'])

    def evaluate(self, batch, stage=None):
        x, (emb_y, logits_y, y) = batch
        preds = self(x)  # expects forward to compute class logits
        loss = F.cross_entropy(preds, y)
        if preds.shape[1] == 100:
            sc_logits = F.max_pool1d(preds[:, cifar100_idcs].unsqueeze(1), kernel_size=5, stride=5)
            sc_pred = torch.argmax(sc_logits.squeeze(), dim=1)
            logits_y = F.one_hot(y, num_classes=100).float()
            sc_logits_y = F.max_pool1d(logits_y[:, cifar100_idcs].unsqueeze(1), kernel_size=5, stride=5)
            sc_y = torch.argmax(sc_logits_y.squeeze(), dim=1)
            sc_acc = accuracy(sc_pred, sc_y)
            self.log(f"{stage}_superclass_acc", sc_acc, prog_bar=True, on_step=False, on_epoch=True)
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
            steps_per_epoch = self.hparams.steps_per_epoch
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
    def __init__(self, class_embeddings_tensor, loss, normalize=False, test_classes=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'normalize', 'test_classes'])
        self.model = my_create_model(class_embeddings_tensor.shape[1])
        self.class_embeddings_tensor = class_embeddings_tensor
        self.loss = loss
        self.normalize = normalize
        self.test_classes = test_classes
        if self.test_classes is not None:
            mask = np.zeros(self.class_embeddings_tensor.shape[0])
            mask[self.test_classes] = 1
            mask = mask[:, np.newaxis]
            self.test_mask = torch.tensor(mask, device=self.class_embeddings_tensor.device, dtype=torch.float32)

    def forward(self, x):
        if self.test_classes is not None and self.stage == 'test':
            class_embeddings = self.test_mask * self.class_embeddings_tensor
        else:
            class_embeddings = self.class_embeddings_tensor
        embedding = self.model(x)
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        logits = embedding @ class_embeddings.T
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


class OurProjectionLitResnet(BaseModule):
    def __init__(self, class_embeddings_tensor, loss, normalize=False, test_classes=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'normalize', 'test_classes'])
        self.model = create_projection_model() # this outputs feature vectors of length 512
        self.projector = nn.Linear(class_embeddings_tensor.shape[1], 512)
        self.class_embeddings_tensor = class_embeddings_tensor
        self.loss = loss
        self.normalize = normalize
        self.test_classes = test_classes
        if self.test_classes is not None:
            mask = np.zeros(self.class_embeddings_tensor.shape[0])
            mask[self.test_classes] = 1
            mask = mask[:, np.newaxis]
            self.test_mask = torch.tensor(mask, device=self.class_embeddings_tensor.device, dtype=torch.float32)

    def forward(self, x):
        if self.test_classes is not None and self.stage == 'test':
            class_embeddings = self.test_mask * self.class_embeddings_tensor
        else:
            class_embeddings = self.class_embeddings_tensor
        class_embeddings = self.projector(class_embeddings)
        embedding = self.model(x)
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        logits = embedding @ class_embeddings.T
        return logits

    def training_step(self, batch, batch_idx):
        x, (emb_y, logits_y, y) = batch
        assert self.model.training
        embedding = self.model(x)
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        loss = self.loss(embedding, self.projector(emb_y), logits_y, y, self.projector(self.class_embeddings_tensor))
        self.log("train_loss", loss)
        return loss


class LitResnet(BaseModule):
    def __init__(self, n_classes=10, temperature=0., **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(n_classes)
        self.temperature = temperature

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, (emb_y, logits_y, y) = batch
        if self.temperature == 0:
            loss = F.cross_entropy(self(x), y)
        else:
            loss = F.cross_entropy(self(x), F.softmax(logits_y / self.temperature, dim=1))
        self.log("train_loss", loss)
        return loss






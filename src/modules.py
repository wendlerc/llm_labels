import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy


def create_model(n_outputs):
    model = torchvision.models.resnet18(pretrained=False, num_classes=n_outputs)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class OurLitResnet(LightningModule):
    def __init__(self, class_embeddings_tensor, loss, three_phase=True, pct_start=0.1, lr=0.05, max_lr=0.1, batch_size=256, weight_decay=5e-4, momentum=0.9, n_train=45000):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor'])
        self.model = create_model(class_embeddings_tensor.shape[1])
        self.class_embeddings_tensor = class_embeddings_tensor
        self.n_train = n_train
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert self.model.training
        embedding = self(x)
        target = y
        loss = self.loss(embedding, target)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        assert not self.model.training
        embedding = self(x)
        # embedding = F.normalize(embedding)# go onto unit sphere where gpt embeddings live
        target = y
        preds = torch.argmax(embedding @ self.class_embeddings_tensor.T, dim=1)
        true_label = torch.argmax(y @ self.class_embeddings_tensor.T, dim=1)
        loss = self.loss(embedding, target)
        acc = accuracy(preds, true_label)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = self.n_train // self.hparams.batch_size
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class LitResnet(LightningModule):
    def __init__(self, n_classes=10, three_phase=True, pct_start=0.1, lr=0.05, max_lr=0.1, batch_size=256, weight_decay=5e-4, momentum=0.9, n_train=45000):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(n_classes)
        self.n_train = n_train

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        preds = self(x)  # go onto unit sphere where gpt embeddings live
        loss = F.cross_entropy(preds, y)
        acc = accuracy(torch.argmax(preds, dim=1), y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = self.n_train // self.hparams.batch_size
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

class OurLitResnetWithProjectorDoesNotWork(LightningModule):
    def __init__(self, class_embeddings_tensor, loss, project_dim=None, three_phase=True, pct_start=0.1, lr=0.05, max_lr=0.1, batch_size=256, weight_decay=5e-4, momentum=0.9, n_train=45000):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'class_embeddings_tensor', 'project_dim'])
        if project_dim is None:
            self.model = create_model(class_embeddings_tensor.shape[1])
            self.projector = None
        else:
            self.model = create_model(project_dim)
            self.projector = nn.Linear(class_embeddings_tensor.shape[1], project_dim)
            #for _, param in self.projector.named_parameters():
            #    param.requires_grad = False
        self.class_embeddings_tensor = class_embeddings_tensor
        self.n_train = n_train
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert self.model.training
        embedding = self(x)
        if self.projector is not None:
            target = self.projector(y)
        else:
            target = y
        loss = self.loss(embedding, target)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        assert not self.model.training
        embedding = self(x)
        # embedding = F.normalize(embedding)# go onto unit sphere where gpt embeddings live
        if self.projector is not None:
            target = self.projector(y)
            class_embeddings = self.projector(self.class_embeddings_tensor)
            preds = torch.argmax(embedding @ class_embeddings.T, dim=1)
            true_label = torch.argmax(target @ class_embeddings.T, dim=1)
        else:
            target = y
            preds = torch.argmax(embedding @ self.class_embeddings_tensor.T, dim=1)
            true_label = torch.argmax(y @ self.class_embeddings_tensor.T, dim=1)
        loss = self.loss(embedding, target)
        acc = accuracy(preds, true_label)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = self.n_train // self.hparams.batch_size
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
from argparse import ArgumentParser

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
import json


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
import yaml
import sys
from shutil import copyfile


def create_model(n_outputs):
    model = torchvision.models.resnet18(pretrained=False, num_classes=n_outputs)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class OurLitResnet(LightningModule):
    def __init__(self, class_embeddings_tensor, lr=0.05, max_lr=0.1, batch_size=256, weight_decay=5e-4, momentum=0.9, n_train=45000):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(class_embeddings_tensor.shape[1])
        self.class_embeddings_tensor = class_embeddings_tensor
        self.n_train = n_train

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        embedding = F.normalize(self(x)) # go onto unit sphere where gpt embeddings live
        loss = F.mse_loss(embedding, y, reduction='sum')
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        embedding = F.normalize(self(x)) # go onto unit sphere where gpt embeddings live
        loss = F.mse_loss(embedding, y, reduction='sum')
        preds = torch.argmax(embedding @ self.class_embeddings_tensor.T, dim=1)
        true_label = torch.argmax(y @ self.class_embeddings_tensor.T, dim=1)
        acc = accuracy(preds, true_label)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.stack([o['loss'] for o in outputs]))
        self.log('mean_val_loss', mean_loss)
        mean_acc = torch.mean(torch.stack([o['acc'] for o in outputs]))
        self.log('mean_val_acc', mean_acc)

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def test_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.stack([o['loss'] for o in outputs]))
        self.log('mean_test_loss', mean_loss)
        mean_acc = torch.mean(torch.stack([o['acc'] for o in outputs]))
        self.log('mean_test_acc', mean_acc)

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
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    # wandb args
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_project', default='lm_labels_cifar10', type=str)
    parser.add_argument('--wandb_entity', default='chrisxx', type=str)
    parser.add_argument('--checkpoint_yaml', default='checkpoint_callback.yaml')
    parser.add_argument('--group', default=None)# this is useful to organize the runs
    # datamodule args
    parser.add_argument('--embedding_file', default='embeddings/cifar_10_davinci-001.json')
    parser.add_argument('--data_path', default='data/datasets/', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    # lightingmodule args
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--max_lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    # trainer args
    parser.add_argument('--monitor', type=str, default='mean_val_loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=2)
    parser.add_argument('--early_stopping_mode', type=str, default='min')
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    parser.add_argument('--my_log_every_n_steps', type=int, default=1)
    parser.add_argument('--my_accelerator', type=str, default='gpu')
    parser.add_argument('--my_max_epochs', type=int, default=200)
    parser.add_argument('--upload', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)    
    # ------------
    # data
    # ------------
    with open(args.embedding_file, 'r') as f:
        class_embeddings = json.load(f)

    classids = {'airplane': 0,
                'automobile': 1,
                'bird': 2,
                'cat': 3,
                'deer': 4,
                'dog': 5,
                'frog': 6,
                'horse': 7,
                'ship': 8,
                'truck': 9}
    classlabels = {value: key for key, value in classids.items()}

    def target_transform(target):
        return torch.tensor(class_embeddings[classlabels[target]])

    dataset = CIFAR10(args.data_path, target_transform=target_transform, download=True)
    n_outputs = dataset[0][1].shape[0]

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    cifar10_dm.EXTRA_ARGS['target_transform'] = target_transform

    # ------------
    # model
    # ------------
    class_embeddings_tensor = torch.tensor([class_embeddings[label] for idx, label in classlabels.items()])
    if args.my_accelerator == 'gpu':
        class_embeddings_tensor = class_embeddings_tensor.to('cuda:0')
    model = OurLitResnet(class_embeddings_tensor,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay,
                         batch_size=args.batch_size)

    # ------------
    # wandb 
    # ------------
    wandb_logger = WandbLogger(entity=args.wandb_entity, 
                               project=args.wandb_project, 
                               name=args.wandb_name,
                               config=args)
    run = wandb_logger.experiment
    # save file to artifact folder
    result_dir = args.checkpoint_dir+'/%s/'%wandb_logger.experiment.name 
    os.makedirs(result_dir, exist_ok=True)
    copyfile(sys.argv[0], result_dir+sys.argv[0].split('/')[-1])
        
    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir+'/%s'%wandb_logger.experiment.name, 
                                          save_top_k=args.checkpoint_save_top_k,
                                          monitor=args.monitor)
    es_callback = EarlyStopping(monitor=args.monitor, 
                                mode=args.early_stopping_mode, 
                                patience=args.early_stopping_patience)
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback,
                                                       es_callback, lr_monitor,
                                                       TQDMProgressBar(refresh_rate=10)],
                                            log_every_n_steps=args.my_log_every_n_steps,
                                            accelerator=args.my_accelerator,
                                            max_epochs=args.my_max_epochs)

    trainer.fit(model, cifar10_dm)
    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=cifar10_dm, ckpt_path='best')

    print(result)
    if args.upload:
        print("uploading model...")
        #store config and model
        checkpoint_callback.to_yaml(checkpoint_callback.dirpath+'/checkpoint_callback.yaml')
        with open(checkpoint_callback.dirpath+'/config.yaml', 'w') as f:
            yaml.dump(run.config.as_dict(), f, default_flow_style=False)
        
        trained_model_artifact = wandb.Artifact(run.name, type="model", description="trained OurLitResnet")
        trained_model_artifact.add_dir(checkpoint_callback.dirpath)
        run.log_artifact(trained_model_artifact)


if __name__ == '__main__':
    main()


import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
import yaml
import sys
from shutil import copyfile
from argparse import ArgumentParser
import os

from utils import get_our_module_and_dataloader, get_baseline_module_and_dataloader

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    # wandb args
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_project', default='lm_labels_cifar10', type=str)
    parser.add_argument('--wandb_entity', default='dfstransformer', type=str)
    parser.add_argument('--checkpoint_yaml', default='checkpoint_callback.yaml')
    parser.add_argument('--group', default=None)# this is useful to organize the runs
    # datamodule args
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--embedding_file_pattern', default='embeddings/%s_davinci-001.json')
    parser.add_argument('--data_path', default='data/datasets/', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    # lightingmodule args
    parser.add_argument('--method', default='ours', type=str)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--max_lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--pct_start', default=0.3, type=float)
    parser.add_argument('--three_phase', default=False, type=bool)
    parser.add_argument('--loss', default='mse', type=str)
    # trainer args
    parser.add_argument('--monitor', type=str, default='val_loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=2)
    parser.add_argument('--early_stopping_mode', type=str, default='min')
    parser.add_argument('--early_stopping_patience', type=int, default=25)
    parser.add_argument('--my_log_every_n_steps', type=int, default=1)
    parser.add_argument('--my_accelerator', type=str, default='gpu')
    parser.add_argument('--my_max_epochs', type=int, default=200)
    parser.add_argument('--upload', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.method == 'ours':
        model, datamodule = get_our_module_and_dataloader(args)
    elif args.method == 'baseline':
        model, datamodule = get_baseline_module_and_dataloader(args)
    else:
        raise ValueError("unrecognized option %s for --method, please use 'ours' or 'baseline'" % args.method)
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
                                          monitor=args.monitor,
                                          save_on_train_epoch_end=False)

    es_callback = EarlyStopping(monitor=args.monitor,
                                mode=args.early_stopping_mode,
                                patience=args.early_stopping_patience,
                                check_on_train_epoch_end=False)
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback,
                                                       es_callback, lr_monitor,
                                                       TQDMProgressBar(refresh_rate=10)],
                                            log_every_n_steps=args.my_log_every_n_steps,
                                            accelerator=args.my_accelerator,
                                            max_epochs=args.my_max_epochs)

    trainer.fit(model, datamodule)
    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=datamodule, ckpt_path='best')

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

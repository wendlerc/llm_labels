
from loss import OutputCE, OutputMSE, OutputCosLoss
from data import get_cifar_datamodule
from output_embeddings import get_cifar_output_embeddings
from modules import OurLitResnet, LitResnet, OurProjectionLitResnet
import torch
import numpy as np


def get_module(args, class_embeddings_tensor, steps_per_epoch):
    if args.method == 'ours' or args.method == 'projection':
        if args.loss == 'emb_mse':
            loss = OutputMSE(reduction=args.loss_reduction)
        elif args.loss == 'emb_ce':
            loss = OutputCE(class_embeddings_tensor, temperature=args.softmax_temperature)
        elif args.loss == 'emb_cos':
            loss = OutputCosLoss(reduction=args.loss_reduction)
        else:
            raise ValueError("unrecognized option %s for --loss." % args.loss)

        if args.dataset == 'cifar100_zeroshot':
            test_classes = np.arange(0, 100, 5)
        else:
            test_classes = None

        if args.method == 'ours':
            model = OurLitResnet(class_embeddings_tensor,
                                 normalize=args.normalize,
                                 loss=loss,
                                 scheduler=args.scheduler,
                                 optimizer=args.optimizer,
                                 pct_start=args.pct_start,
                                 three_phase=args.three_phase,
                                 lr=args.lr,
                                 max_lr=args.max_lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 batch_size=args.batch_size,
                                 steps_per_epoch=steps_per_epoch,
                                 test_classes=test_classes)
        elif args.method == 'projection':
            model = OurProjectionLitResnet(class_embeddings_tensor,
                                 normalize=args.normalize,
                                 loss=loss,
                                 scheduler=args.scheduler,
                                 optimizer=args.optimizer,
                                 pct_start=args.pct_start,
                                 three_phase=args.three_phase,
                                 lr=args.lr,
                                 max_lr=args.max_lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 batch_size=args.batch_size,
                                 steps_per_epoch=steps_per_epoch,
                                 test_classes=test_classes)
        else:
            raise ValueError("unrecognized option %s for --method." % args.method)

    elif args.method == 'baseline':
        model = LitResnet(n_classes=class_embeddings_tensor.shape[0],
                          temperature=args.softmax_temperature,
                          scheduler=args.scheduler,
                          optimizer=args.optimizer,
                          pct_start=args.pct_start,
                          three_phase=args.three_phase,
                          lr=args.lr,
                          max_lr=args.max_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          batch_size=args.batch_size,
                          steps_per_epoch=steps_per_epoch)
    return model


def get_datamodule_and_classembeddings(args):
    # ------------
    # data
    # ------------
    class_embeddings, classids, classlabels, class_embeddings_tensor = get_cifar_output_embeddings(args)
    logits = class_embeddings_tensor.cpu() @ class_embeddings_tensor.T.cpu()

    def target_transform(target):
        return torch.tensor(class_embeddings[classlabels[target]]), logits[target], target

    datamodule = get_cifar_datamodule(args, target_transform=target_transform)
    return datamodule, class_embeddings_tensor


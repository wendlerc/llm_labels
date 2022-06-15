
from loss import OutputCE, OutputMSE
from data import get_cifar10_datamodule, get_cifar100_datamodule
from output_embeddings import get_cifar10_output_embeddings, get_cifar100_output_embeddings
from modules import OurLitResnet, LitResnet
import torch


def get_our_module_and_dataloader(args):
    # ------------
    # data
    # ------------
    if args.dataset == 'cifar10':
        class_embeddings, classids, classlabels, class_embeddings_tensor = get_cifar10_output_embeddings(args)

        def target_transform(target):
            return torch.tensor(class_embeddings[classlabels[target]]), target

        datamodule = get_cifar10_datamodule(args, target_transform=target_transform)
    elif args.dataset == 'cifar100':
        class_embeddings, classids, classlabels, class_embeddings_tensor = get_cifar100_output_embeddings(args)

        def target_transform(target):
            return torch.tensor(class_embeddings[classlabels[target]]), target

        datamodule = get_cifar100_datamodule(args, target_transform=target_transform)
    else:
        raise ValueError("unrecognized option %s for --dataset, please use 'cifar10' or 'cifar100'" % args.loss)
    # ------------
    # model
    # ------------
    if args.loss == 'mse':
        loss = OutputMSE()
    elif args.loss == 'ce':
        loss = OutputCE(class_embeddings_tensor)
    else:
        raise ValueError("unrecognized option %s for --loss, please use 'mse' or 'ce'" % args.loss)

    model = OurLitResnet(class_embeddings_tensor,
                         loss=loss,
                         scheduler=args.scheduler,
                         optimizer=args.optimizer,
                         pct_start=args.pct_start,
                         three_phase=args.three_phase,
                         lr=args.lr,
                         max_lr=args.max_lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay,
                         batch_size=args.batch_size)
    return model, datamodule


def get_baseline_module_and_dataloader(args):
    # ------------
    # data
    # ------------
    def target_transform(target):
        return [], target

    if args.dataset == 'cifar10':
        datamodule = get_cifar10_datamodule(args, target_transform)
        n_classes = 10
    elif args.dataset == 'cifar100':
        datamodule = get_cifar100_datamodule(args, target_transform)
        n_classes = 100
    else:
        raise ValueError("unrecognized option %s for --dataset, please use 'cifar10' or 'cifar100'" % args.loss)
    # ------------
    # model
    # ------------
    model = LitResnet(n_classes=n_classes,
                      scheduler=args.scheduler,
                      pct_start=args.pct_start,
                      three_phase=args.three_phase,
                      lr=args.lr,
                      max_lr=args.max_lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      batch_size=args.batch_size)
    return model, datamodule


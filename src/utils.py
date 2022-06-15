
from loss import OutputCE, OutputMSE, OutputCosLoss
from data import get_cifar10_datamodule, get_cifar100_datamodule
from output_embeddings import get_cifar10_output_embeddings, get_cifar100_output_embeddings
from modules import OurLitResnet, LitResnet
import torch


def get_module(args, class_embeddings_tensor):
    if args.method == 'ours':
        if args.loss == 'emb_mse':
            loss = OutputMSE()
        elif args.loss == 'emb_ce':
            loss = OutputCE(class_embeddings_tensor, temperature=args.softmax_temperature)
        elif args.loss == 'emb_cos':
            loss = OutputCosLoss()
        else:
            raise ValueError("unrecognized option %s for --loss." % args.loss)

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
                             batch_size=args.batch_size)
    elif args.method == 'baseline':
        model = LitResnet(n_classes=class_embeddings_tensor.shape[0],
                          temperature=args.softmax_temperature,
                          scheduler=args.scheduler,
                          pct_start=args.pct_start,
                          three_phase=args.three_phase,
                          lr=args.lr,
                          max_lr=args.max_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          batch_size=args.batch_size)
    return model


def get_datamodule_and_classembeddings(args):
    # ------------
    # data
    # ------------
    if args.dataset == 'cifar10':
        class_embeddings, classids, classlabels, class_embeddings_tensor = get_cifar10_output_embeddings(args)
        logits = class_embeddings_tensor.cpu() @ class_embeddings_tensor.T.cpu()

        def target_transform(target):
            return torch.tensor(class_embeddings[classlabels[target]]), logits[target], target

        datamodule = get_cifar10_datamodule(args, target_transform=target_transform)
    elif args.dataset == 'cifar100':
        class_embeddings, classids, classlabels, class_embeddings_tensor = get_cifar100_output_embeddings(args)
        logits = class_embeddings_tensor.cpu() @ class_embeddings_tensor.T.cpu()

        def target_transform(target):
            return torch.tensor(class_embeddings[classlabels[target]]), logits[target], target

        datamodule = get_cifar100_datamodule(args, target_transform=target_transform)
    else:
        raise ValueError("unrecognized option %s for --dataset, please use 'cifar10' or 'cifar100'" % args.loss)

    return datamodule, class_embeddings_tensor


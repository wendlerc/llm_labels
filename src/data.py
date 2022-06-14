import torch
import torchvision
from torchvision.datasets import CIFAR100
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


def get_cifar10_datamodule(class_embeddings, classlabels, args):
    def target_transform(target):
        return torch.tensor(class_embeddings[classlabels[target]])

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
    return cifar10_dm


def get_cifar100_datamodule(class_embeddings, classlabels, args):
    def target_transform(target):
        return torch.tensor(class_embeddings[classlabels[target]])

    dataset = CIFAR100(args.data_path, target_transform=target_transform, download=True)
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

    cifar100_dm = CIFAR10DataModule(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    cifar100_dm.name = 'cifar100'
    cifar100_dm.dataset_cls = CIFAR100
    cifar100_dm.EXTRA_ARGS['target_transform'] = target_transform
    return cifar100_dm

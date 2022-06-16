import torch
import torchvision
from torchvision.datasets import CIFAR100, VisionDataset
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import numpy as np


class CIFAR100ZeroShot(VisionDataset):
    """ CIFAR100 with a different split that allows for zero-shot learning evaluation."""
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 test_classes=np.arange(0, 100, 6)):
        super(CIFAR100ZeroShot, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.cifar100_train = CIFAR100(root, train=True, download=download, transform=transform,
                                       target_transform=target_transform)
        self.cifar100_test = CIFAR100(root, train=False, download=download, transform=transform,
                                      target_transform=target_transform)
        self.test_classes = test_classes

        if train:
            self.data = [item for item in self.cifar100_train if item[-1] not in test_classes]
            self.data += [item for item in self.cifar100_test if item[-1] not in test_classes]
        else:
            self.data = [item for item in self.cifar100_train if item[-1] in test_classes]
            self.data += [item for item in self.cifar100_test if item[-1] in test_classes]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_cifar_datamodule(args, target_transform=None):
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

    cifar_dm = CIFAR10DataModule(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    if target_transform is not None:
        cifar_dm.EXTRA_ARGS['target_transform'] = target_transform
    if args.dataset == 'cifar100':
        cifar_dm.name = 'cifar100'
        cifar_dm.dataset_cls = CIFAR100
    elif args.dataset == 'cifar100_zeroshot':
        cifar_dm.name = 'cifar100_zeroshot'
        cifar_dm.dataset_cls = CIFAR100ZeroShot
    elif args.dataset == 'cifar10':
        pass
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    return cifar_dm




import json
import torch


def get_cifar_output_embeddings(args):
    if args.dataset == 'cifar10':
        embedding_file = 'embeddings/%s_%s-001.json'%(args.dataset, args.embeddings)
        with open(embedding_file, 'r') as f:
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
        class_embeddings_tensor = torch.tensor([class_embeddings[label] for idx, label in classlabels.items()])
        if args.accelerator == 'gpu':
            class_embeddings_tensor = class_embeddings_tensor.to('cuda:0')
        return class_embeddings, classids, classlabels, class_embeddings_tensor
    elif args.dataset == 'cifar100' or args.dataset == 'cifar100_zeroshot':
        embedding_file = 'embeddings/cifar100_%s-001.json' % (args.embeddings)
        with open(embedding_file, 'r') as f:
            class_embeddings = json.load(f)

        labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                  'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                  'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                  'bottles', 'bowls', 'cans', 'cups', 'plates',
                  'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                  'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                  'bed', 'chair', 'couch', 'table', 'wardrobe',
                  'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                  'bear', 'leopard', 'lion', 'tiger', 'wolf',
                  'bridge', 'castle', 'house', 'road', 'skyscraper',
                  'cloud', 'forest', 'mountain', 'plain', 'sea',
                  'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                  'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                  'crab', 'lobster', 'snail', 'spider', 'worm',
                  'baby', 'boy', 'girl', 'man', 'woman',
                  'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                  'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                  'maple', 'oak', 'palm', 'pine', 'willow',
                  'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                  'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

        classids = {l: idx for idx, l in enumerate(labels)}
        classlabels = {idx: l for idx, l in enumerate(labels)}
        class_embeddings_tensor = torch.tensor([class_embeddings[label] for idx, label in classlabels.items()])
        if args.accelerator == 'gpu':
            class_embeddings_tensor = class_embeddings_tensor.to('cuda:0')
        return class_embeddings, classids, classlabels, class_embeddings_tensor
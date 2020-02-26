from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.PARLoader import PARDataset, PARTestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
imagenet_datasets = ["imagenet", "imagenet-sketch"]
available_datasets = pacs_datasets + imagenet_datasets


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    if dataset_list[0] in pacs_datasets:
        for dname in dataset_list:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
            train_dataset = PARDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                          tile_transformer=tile_transformer)
            if limit:
                train_dataset = Subset(train_dataset, limit)
            datasets.append(train_dataset)
            val_datasets.append(
                PARTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                                  patches=patches))
    elif dataset_list[0] in imagenet_datasets:
        name_train, labels_train = _dataset_info(join(join(dirname(__file__), 'txt_lists', 'imagenet_trainDataPath.txt')))
        train_dataset = PARDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer)
        datasets.append(train_dataset)
        name_val, labels_val = _dataset_info(join(join(dirname(__file__), 'txt_lists', 'imagenet_valDataPath.txt')))
        val_datasets.append(
            PARTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                              patches=patches))
    else:
        print('Error: dataset not found.')
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args, patches=False):
    if args.target in pacs_datasets:
        names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    elif args.target in imagenet_datasets:
        names, labels = _dataset_info(join(join(join(dirname(__file__), 'txt_lists', 'imagenet_testDataPath.txt'))))
    else:
        print('Error: test dataset not found.')
    img_tr = get_val_transformer(args)
    val_dataset = PARTestDataset(names, labels, patches=patches, img_transformer=img_tr)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_par_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = [transforms.Resize((args.image_size, args.image_size))]
    tile_tr = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    img_transformer = transforms.Compose(img_tr)
    tile_transformer = transforms.Compose(tile_tr)
    val_dataset = PARDataset(names, labels, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_target_par_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = PARDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader

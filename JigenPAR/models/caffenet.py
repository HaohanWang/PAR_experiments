import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from models.alexnet import Id
from models.model_utils import ReverseLayerF

class AlexNetCaffe_PAR_H(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe_PAR_H, self).__init__()
        print("Using Caffe AlexNet")
        self.features_local = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
        ]))
        self.features = nn.Sequential(OrderedDict([
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        # self.class_classifier = nn.Linear(4096, n_classes)
        self.class_classifier = nn.Sequential(OrderedDict([("fc8", nn.Linear(4096, n_classes))]))

        self.PAR_classifier = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

    def get_params(self):
        return [{"params": chain(self.features_local.parameters(), self.classifier.parameters(),
                self.features.parameters(), self.class_classifier.parameters())}]
    def get_par_params(self):
        return [{"params": self.PAR_classifier.parameters()}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x_local = self.features_local(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.features(x_local)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x), self.PAR_classifier(x_local).permute(0, 2, 3, 1).contiguous()

class AlexNetCaffe_PAR_B(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe_PAR_B, self).__init__()
        print("Using Caffe AlexNet")
        self.features_local = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
        ]))
        self.features = nn.Sequential(OrderedDict([
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        # self.class_classifier = nn.Linear(4096, n_classes)
        self.class_classifier = nn.Sequential(OrderedDict([("fc8", nn.Linear(4096, n_classes))]))

        self.PAR_classifier = nn.Conv2d(96, n_classes, kernel_size=3, stride=1)

    def get_params(self):
        return [{"params": chain(self.features_local.parameters(), self.classifier.parameters(),
                self.features.parameters(), self.class_classifier.parameters())}]
    def get_par_params(self):
        return [{"params": self.PAR_classifier.parameters()}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x_local = self.features_local(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.features(x_local)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x), self.PAR_classifier(x_local).permute(0, 2, 3, 1).contiguous()

class AlexNetCaffe_PAR_M(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe_PAR_M, self).__init__()
        print("Using Caffe AlexNet")
        self.features_local = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
        ]))
        self.features = nn.Sequential(OrderedDict([
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        # self.class_classifier = nn.Linear(4096, n_classes)
        self.class_classifier = nn.Sequential(OrderedDict([("fc8", nn.Linear(4096, n_classes))]))

        self.PAR_classifier = nn.Sequential(
            nn.Conv2d(96, 100, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 50, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, n_classes, kernel_size=1, stride=1))

    def get_params(self):
        return [{"params": chain(self.features_local.parameters(), self.classifier.parameters(),
                self.features.parameters(), self.class_classifier.parameters())}]
    def get_par_params(self):
        return [{"params": self.PAR_classifier.parameters()}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x_local = self.features_local(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.features(x_local)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x), self.PAR_classifier(x_local).permute(0, 2, 3, 1).contiguous()

class AlexNetCaffe_PAR(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe_PAR, self).__init__()
        print("Using Caffe AlexNet")
        self.features_local = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
        ]))
        self.features = nn.Sequential(OrderedDict([
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        # self.class_classifier = nn.Linear(4096, n_classes)
        self.class_classifier = nn.Sequential(OrderedDict([("fc8", nn.Linear(4096, n_classes))]))

        self.PAR_classifier = nn.Conv2d(96, n_classes, kernel_size=1, stride=1)

    def get_params(self):
        return [{"params": chain(self.features_local.parameters(), self.classifier.parameters(),
                self.features.parameters(), self.class_classifier.parameters())}]
    def get_par_params(self):
        return [{"params": self.PAR_classifier.parameters()}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x_local = self.features_local(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = self.features(x_local)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.class_classifier(x), self.PAR_classifier(x_local).permute(0, 2, 3, 1).contiguous()

class AlexNetCaffe(nn.Module):
    def __init__(self, n_classes=100, domains=3, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))
        self.class_classifier = nn.Sequential(OrderedDict([("fc8", nn.Linear(4096, n_classes))]))

        self.par_classifier = nn.Linear(4096, n_classes)
        # self.class_classifier = nn.Linear(4096, n_classes)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, domains))

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.par_classifier.parameters()
                                 , self.class_classifier.parameters()#, self.domain_classifier.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        #d = ReverseLayerF.apply(x, lambda_val)
        x = self.classifier(x)
        return self.par_classifier(x), self.class_classifier(x)#, self.domain_classifier(d)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AlexNetCaffeAvgPool(AlexNetCaffe):
    def __init__(self, n_classes=100):
        super().__init__()
        print("Global Average Pool variant")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            #             ("relu5", nn.ReLU(inplace=True)),
            #             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))

        self.par_classifier = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(128 * 6 * 6, n_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(1024, n_classes, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(13),
            Flatten(),
            # nn.Linear(1024, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AlexNetCaffeFC7(AlexNetCaffe):
    def __init__(self, n_classes=100, dropout=True):
        super(AlexNetCaffeFC7, self).__init__()
        print("FC7 branching variant")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id())]))

        self.par_classifier = nn.Sequential(OrderedDict([
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout()),
            ("fc8", nn.Linear(4096, n_classes))]))
        self.class_classifier = nn.Sequential(OrderedDict([
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout()),
            ("fc8", nn.Linear(4096, n_classes))]))


def caffenet(classes):
    model = AlexNetCaffe(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    if model.class_classifier.fc8.out_features == state_dict["classifier.fc8.weight"].shape[0]: # imagenet dataset load last layer
        state_dict["class_classifier.fc8.weight"] = state_dict["classifier.fc8.weight"]
        state_dict["class_classifier.fc8.bias"] = state_dict["classifier.fc8.bias"]
    else:
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model


def parnet(classes):
    model = AlexNetCaffe_PAR(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
    for m in model.PAR_classifier.modules():
        nn.init.xavier_uniform_(m.weight, .01)
        nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    # import ipdb;ipdb.set_trace()
    # params = [p for p in model.features_local.parameters()]
    # params2 = [p for p in model.features.parameters()]
    state_dict["features_local.conv1.weight"] = state_dict["features.conv1.weight"] 
    state_dict["features_local.conv1.bias"] = state_dict["features.conv1.bias"] 
    if model.class_classifier.fc8.out_features == state_dict["classifier.fc8.weight"].shape[0]: # imagenet dataset load last layer
        state_dict["class_classifier.fc8.weight"] = state_dict["classifier.fc8.weight"]
        state_dict["class_classifier.fc8.bias"] = state_dict["classifier.fc8.bias"]
    else:
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model

def parnet_B(classes):
    model = AlexNetCaffe_PAR_B(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
    for m in model.PAR_classifier.modules():
        nn.init.xavier_uniform_(m.weight, .01)
        nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    state_dict["features_local.conv1.weight"] = state_dict["features.conv1.weight"] 
    state_dict["features_local.conv1.bias"] = state_dict["features.conv1.bias"] 
    if model.class_classifier.fc8.out_features == state_dict["classifier.fc8.weight"].shape[0]: # imagenet dataset load last layer
        state_dict["class_classifier.fc8.weight"] = state_dict["classifier.fc8.weight"]
        state_dict["class_classifier.fc8.bias"] = state_dict["classifier.fc8.bias"]
    else:
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model

def parnet_M(classes):
    model = AlexNetCaffe_PAR_M(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
    for m in model.PAR_classifier.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, .01)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    state_dict["features_local.conv1.weight"] = state_dict["features.conv1.weight"] 
    state_dict["features_local.conv1.bias"] = state_dict["features.conv1.bias"] 
    if model.class_classifier.fc8.out_features == state_dict["classifier.fc8.weight"].shape[0]: # imagenet dataset load last layer
        state_dict["class_classifier.fc8.weight"] = state_dict["classifier.fc8.weight"]
        state_dict["class_classifier.fc8.bias"] = state_dict["classifier.fc8.bias"]
    else:
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model

def parnet_H(classes):
    model = AlexNetCaffe_PAR_H(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)
    for m in model.PAR_classifier.modules():
        nn.init.xavier_uniform_(m.weight, .01)
        nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    state_dict["features_local.conv1.weight"] = state_dict["features.conv1.weight"] 
    state_dict["features_local.conv1.bias"] = state_dict["features.conv1.bias"] 
    state_dict["features_local.conv2.weight"] = state_dict["features.conv2.weight"] 
    state_dict["features_local.conv2.bias"] = state_dict["features.conv2.bias"] 
    if model.class_classifier.fc8.out_features == state_dict["classifier.fc8.weight"].shape[0]: # imagenet dataset load last layer
        state_dict["class_classifier.fc8.weight"] = state_dict["classifier.fc8.weight"]
        state_dict["class_classifier.fc8.bias"] = state_dict["classifier.fc8.bias"]
    else:
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model
def caffenet_gap(classes):
    model = AlexNetCaffe(classes)
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc6.weight"]
    del state_dict["classifier.fc6.bias"]
    del state_dict["classifier.fc7.weight"]
    del state_dict["classifier.fc7.bias"]
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    # weights are initialized in the constructor
    return model


def caffenet_fc7(classes):
    model = AlexNetCaffeFC7(classes)
    state_dict = torch.load("models/pretrained/alexnet_caffe.pth.tar")
    state_dict["par_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
    state_dict["par_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
    state_dict["class_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
    state_dict["class_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    del state_dict["classifier.fc7.weight"]
    del state_dict["classifier.fc7.bias"]
    model.load_state_dict(state_dict, strict=False)
    nn.init.xavier_uniform_(model.par_classifier.fc8.weight, .1)
    nn.init.constant_(model.par_classifier.fc8.bias, 0.)
    nn.init.xavier_uniform_(model.class_classifier.fc8.weight, .1)
    nn.init.constant_(model.class_classifier.fc8.bias, 0.)
    return model

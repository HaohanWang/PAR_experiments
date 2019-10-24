import argparse

import os
import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler_PAR
from utils.Logger import Logger
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch par training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--par_learning_rate", "-pl", type=float, default=.01, help="PAR Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--par_weight", type=float, default=0.1, help="Weight for the par loss")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=False, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler, self.optimizer_par, self.scheduler_par = get_optim_and_scheduler_PAR(model, args.epochs, args.learning_rate, args.par_learning_rate,args.train_all, nesterov=args.nesterov)
        self.par_weight = args.par_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
            # import ipdb;ipdb.set_trace()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, _, class_l), _) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            # update par classifier
            self.optimizer_par.zero_grad()
            class_logit, par_logit = self.model(data)
            m, n = par_logit.shape[1], par_logit.shape[2]
            par_class_l = class_l.view(-1, 1, 1, 1).repeat(1, m, n, 1).view(-1)
            par_loss = criterion(par_logit.view(-1, self.n_classes), par_class_l)
            _, par_pred = par_logit.view(-1, self.n_classes).max(dim=1)
            par_loss.backward()
            self.optimizer_par.step()

            # update main classifier
            self.optimizer.zero_grad()
            class_logit, par_logit = self.model(data)
            class_loss = criterion(class_logit, class_l)
            # import ipdb;ipdb.set_trace()
            par_loss2 = criterion(par_logit.view(-1, self.n_classes), par_class_l)
            # top1_correct_pred, top5_correct_pred = self.accuracy(class_logit, class_l, topk=[1,5])
            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss - par_loss2 * self.par_weight 
            # loss = class_loss 
            loss.backward()
            self.optimizer.step()


            self.logger.log(it, len(self.source_loader),
                            {"par": par_loss.item(), "class": class_loss.item()  
                             },
                            # ,"lambda": lambda_val},
                            {"par": torch.sum(par_pred == par_class_l.data).type(torch.FloatTensor)/(m*n),
                             "class": torch.sum(cls_pred == class_l.data).item(),
                             # "top5 class": top5_correct_pred.item(),
                             },
                            data.shape[0])
            # print(time()-begin)
            del loss, class_loss, par_loss, par_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                par_correct, top1_correct_pred, top5_correct_pred = self.do_test(loader)
                par_acc = float(par_correct) / total
                class_top1_acc = float(top1_correct_pred) / total
                class_top5_acc = float(top5_correct_pred) / total
                self.logger.log_test(phase, {"par": par_acc, "class top1": class_top1_acc, "class top5": class_top5_acc})
                self.results[phase+'top1'][self.current_epoch] = class_top1_acc
                self.results[phase+'top5'][self.current_epoch] = class_top5_acc

    def do_test(self, loader):
        par_correct = 0
        # class_correct = 0
        class_correct_top1 = 0
        class_correct_top5 = 0
        domain_correct = 0
        for it, ((data, _, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit, par_logit = self.model(data)
            m, n = par_logit.shape[1], par_logit.shape[2]
            par_class_l = class_l.view(-1, 1, 1, 1).repeat(1, m, n, 1).view(-1)
            _, cls_pred = class_logit.max(dim=1)
            _, par_pred = par_logit.view(-1, self.n_classes).max(dim=1)
            top1_correct_pred, top5_correct_pred = self.accuracy(class_logit, class_l, topk=[1,5])
            # class_correct += torch.sum(cls_pred == class_l.data)
            class_correct_top1 += top1_correct_pred
            class_correct_top5 += top5_correct_pred
            # import ipdb;ipdb.set_trace()
            par_correct += torch.sum(par_pred == par_class_l.data).type(torch.FloatTensor)/(m*n)
        return par_correct, class_correct_top1, class_correct_top5

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"valtop1": torch.zeros(self.args.epochs), "valtop5": torch.zeros(self.args.epochs), 
                        "testtop1": torch.zeros(self.args.epochs), "testtop5": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.scheduler_par.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["valtop1"]
        testtop1_res = self.results["testtop1"]
        testtop5_res = self.results["testtop5"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test top1 acc %g top5 acc %g - best test top1: %g, top5: %g" % (val_res.max(), testtop1_res[idx_best], testtop5_res[idx_best], testtop1_res.max(), testtop5_res.max()))
        self.logger.save_best(testtop1_res[idx_best], testtop1_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

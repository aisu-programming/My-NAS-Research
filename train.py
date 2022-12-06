""" Libraries """
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.manual_seed(torch.initial_seed())
torch_generator = torch.Generator()
torch_generator.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    # torch.cuda.set_per_process_memory_fraction(22.0/24.0, DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(SEED)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True  # type: ignore
else:
    DEVICE = torch.device("cpu")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import utils
from model import Model
from architect import Architect


""" Hyperparameters """
NODE_NUMBER = 128
NODE_DEPTH  = 8
INITIAL_THRESHOLD_OF_CONNECTION = 0.3  # t_c
MAX_THRESHOLD_OF_CONNECTION     = 0.7  # max_t_c
MAX_NUMBER_OF_CONNECTION = 16  # max_n_c >= 1
MIN_NUMBER_OF_CONNECTION = 1   # min_n_c >= 1
EPOCHS = 200
BATCH_SIZE = 256


""" Function """
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# class CIFAR10onGPU(torch.utils.data.Dataset):
#     def __init__(self, root, train, download, transform):
#         super(CIFAR10onGPU, self).__init__()
#         assert DEVICE == torch.device("cuda:0")
#         dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
#         self.dataset = [ (
#             torch.Tensor(dataset.__getitem__(id)[0]).to(DEVICE),
#             torch.Tensor(dataset.__getitem__(id)[1]).to(DEVICE)
#         ) for id in range(len(dataset)) ]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, i):
#         return self.dataset[i]


def main(args):

    """ Misc """
    output_path = args.output_path
    os.makedirs(f"{output_path}", exist_ok=True)

    """ Model """
    criterion = torch.nn.CrossEntropyLoss()
    node_num, node_depth = args.node_number, args.node_depth
    threshold_of_connection, max_number_of_connection, min_number_of_connection = \
        args.t_c, args.max_n_c, args.min_n_c
    batch_size = args.batch_size
    model = Model(node_num, node_depth, threshold_of_connection, max_number_of_connection,
                  min_number_of_connection, criterion, output_path, DEVICE).to(DEVICE)
    # model.search_path(plot_dag=False)
    # raise Exception

    """ Data """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True, transform=transform)
    # train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [ 45000, 5000 ], torch_generator)
    # test_dataset  = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    # test_dataloader  = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    """ Training """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9975)

    """ Train """
    train_iterator = enumerate(train_dataloader.__iter__())
    # valid_iterator = enumerate(valid_dataloader.__iter__())
    for epoch in range(1, args.epochs+1):

        model.search_path()
        if model.t_c+0.005 <= args.max_t_c: model.t_c += 0.005
        node_losses, node_corrects = [], []
        alpha_losses, alpha_corrects = [], []

        pbar = tqdm(total=50, ascii=True)
        pbar_alpha_des, pbar_node_des = "", ""
        for _ in range(50):

            """ Train alphas using valid dataset """
            # model.unfreeze_alphas()
            # model.freeze_nodes()
            try:
                bi, (batch_imgs, batch_labels) = next(train_iterator)
            except StopIteration:
                train_iterator = enumerate(train_dataloader.__iter__())
                bi, (batch_imgs, batch_labels) = next(train_iterator)
            batch_imgs, batch_labels = batch_imgs.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            batch_predictions:torch.Tensor = model(batch_imgs)
            loss = criterion(batch_predictions, batch_labels)
            loss.backward()
            optimizer.step()
            if epoch >= 50: lr_scheduler.step()
            
            alpha_corrects += list((torch.argmax(batch_predictions, dim=-1)==batch_labels).cpu().detach().numpy())
            alpha_losses.append(loss.item())
            pbar.update(1)
            pbar_alpha_des = f"Alphas Avg Loss: {np.average(alpha_losses):.4f}, " + \
                             f"Alphas Acc: {np.average(alpha_corrects)*100:6.3f}%"
            pbar.set_description(f"Epoch {epoch:3}/100 | " + pbar_alpha_des + f" | LR: {get_lr(optimizer):.8f}")

        pbar.close()


def main_DARTS(args):

    """ Misc """
    output_path = args.output_path
    os.makedirs(f"{output_path}", exist_ok=True)

    """ Model """
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)
    node_num, node_depth = args.node_number, args.node_depth
    threshold_of_connection, max_number_of_connection, min_number_of_connection = \
        args.t_c, args.max_n_c, args.min_n_c
    batch_size = args.batch_size
    model = Model(node_num, node_depth, threshold_of_connection, max_number_of_connection,
                  min_number_of_connection, criterion, output_path, DEVICE).to(DEVICE)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    print("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.9 * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.lr_min)  # type: ignore

    architect = Architect(model, args, DEVICE)

    for epoch in range(args.epochs):

        """ Add by me """
        model.search_path()
        if model.t_c+0.005 <= args.max_t_c: model.t_c += 0.005

        lr = get_lr(optimizer)
        # logging.info('epoch %d lr %e', epoch, lr)
        print(f"epoch {epoch}: lr={lr}")

        # genotype = model.genotype()
        # logging.info('genotype = %s', genotype)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        scheduler.step()
        print('train_acc %f', train_acc)
        # logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        print('valid_acc %f', valid_acc)
        # logging.info('valid_acc %f', valid_acc)

        # utils.save(model, os.path.join(args.output_path, 'weights.pt'))
    return

def train(train_queue, valid_queue, model:Model, architect:Architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()

        input  = torch.autograd.Variable(input, requires_grad=False).to(DEVICE)
        target = torch.autograd.Variable(target, requires_grad=False).to(DEVICE)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search  = torch.autograd.Variable(input_search, requires_grad=False).to(DEVICE)
        target_search = torch.autograd.Variable(target_search, requires_grad=False).to(DEVICE)
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        print(f"train {step:03d} {objs.avg:.4f} {top1.avg:5.2f} {top5.avg:5.2f}")

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input  = torch.autograd.Variable(input, volatile=True).to(DEVICE)
        target = torch.autograd.Variable(target, volatile=True).to(DEVICE)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        print(f"valid {step:03d} {objs.avg:.4f} {top1.avg:5.2f} {top5.avg:5.2f}")

    return top1.avg, objs.avg


""" Execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    """ Model """
    parser.add_argument("--node_number", type=int, default=NODE_NUMBER, help="Number of nodes")
    parser.add_argument("--node_depth" , type=int, default=NODE_DEPTH , help="Depth of nodes")
    parser.add_argument("--t_c", type=float, default=INITIAL_THRESHOLD_OF_CONNECTION, help="")
    parser.add_argument("--max_t_c", type=float, default=MAX_THRESHOLD_OF_CONNECTION, help="")
    parser.add_argument("--max_n_c", type=int, default=MAX_NUMBER_OF_CONNECTION, help="Max number of connection")
    parser.add_argument("--min_n_c", type=int, default=MIN_NUMBER_OF_CONNECTION, help="Min number of connection")
    # parser.add_argument("--", action="store_true", help="")
    
    """ Training """
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of nodes")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    
    """ Misc """
    parser.add_argument("--output_path", type=str, help="Root directory path for output",
        default=f"history/{time.strftime('%Y.%m.%d', time.localtime())}")
        # default=f"history/{time.strftime('%Y.%m.%d/%H.%M.%s', time.localtime())}")

    """ Others from DARTS """
    parser.add_argument("--lr_min", type=float, default=0.001, help="min learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--unrolled", action="store_true", default=True, help="use one-step unrolled validation loss")
    parser.add_argument("--alphas_lr", type=float, default=3e-4, help="")
    parser.add_argument("--alphas_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")

    args = parser.parse_args()
    print(args)
    main(args)
    # main_DARTS(args)
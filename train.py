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

from model import Model



""" Hyperparameters """
NODE_NUMBER = 50
MAX_DEPTH   = 8
THRESHOLD_OF_CONNECTION  = 0.25  # t_c
MAX_NUMBER_OF_CONNECTION = 9     # max_n_c >= 1
MIN_NUMBER_OF_CONNECTION = 1     # min_n_c >= 1
BATCH_SIZE = 64


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

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, i):
#         return self.dataset[i]


def main(opt):

    """ Misc """
    output_path = opt.output_path
    os.makedirs(f"{output_path}/DAG_before_prune", exist_ok=True)
    os.makedirs(f"{output_path}/DAG_after_prune", exist_ok=True)

    """ Model """
    node_num, max_depth = opt.node_number, opt.max_depth
    threshold_of_connection = opt.t_c
    max_number_of_connection, min_number_of_connection = opt.max_n_c, opt.min_n_c
    batch_size = opt.batch_size
    model = Model(node_num, max_depth, threshold_of_connection, max_number_of_connection,
                  min_number_of_connection, output_path).to(DEVICE)

    """ Data """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True, transform=transform)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [ 45000, 5000 ], torch_generator)
    test_dataset  = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    test_dataloader  = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    """ Training """
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    """ Train """
    valid_iterator = enumerate(valid_dataloader.__iter__())
    train_iterator = enumerate(train_dataloader.__iter__())
    for epoch in range(1, 100+1):

        model.search_path(plot_dag=True)
        train_losses, train_corrects = [], []
        valid_losses, valid_corrects = [], []

        for _ in range(5):

            """ Train alphas using valid dataset """
            model.unfreeze_alphas()
            model.freeze_nodes()
            # pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), ascii=True)
            # for bi, (batch_imgs, batch_labels) in pbar:
            #     if bi == 50: break
            try:
                bi, (batch_imgs, batch_labels) = next(valid_iterator)
            except StopIteration:
                valid_iterator = enumerate(valid_dataloader.__iter__())
                bi, (batch_imgs, batch_labels) = next(valid_iterator)

            batch_imgs, batch_labels = batch_imgs.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            batch_predictions:torch.Tensor = model(batch_imgs)
            loss = loss_fn(batch_predictions, batch_labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # lr_scheduler.step(loss.item())
            
            valid_corrects += list((torch.argmax(batch_predictions, dim=-1)==batch_labels).cpu().detach().numpy())
            valid_losses.append(loss.item())
            

            """ Train nodes using train dataset """
            model.freeze_alphas()
            model.unfreeze_nodes()
            # pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
            # for bi, (batch_imgs, batch_labels) in pbar:
            #     if bi == 50: break
            try:
                bi, (batch_imgs, batch_labels) = next(train_iterator)
            except StopIteration:
                train_iterator = enumerate(train_dataloader.__iter__())
                bi, (batch_imgs, batch_labels) = next(train_iterator)
            batch_imgs, batch_labels = batch_imgs.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            batch_predictions:torch.Tensor = model(batch_imgs)
            loss = loss_fn(batch_predictions, batch_labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # lr_scheduler.step(loss.item())
            
            train_corrects += list((torch.argmax(batch_predictions, dim=-1)==batch_labels).cpu().detach().numpy())
            train_losses.append(loss.item())

        print(f"Epoch {epoch:3}/100 [Alphas] Avg Loss: {np.average(valid_losses):.4f}, " +
              f"Acc: {np.average(valid_corrects)*100:6.3f}%, LR: {get_lr(optimizer):.8f}")
        print(f"Epoch {epoch:3}/100 [Nodes ] Avg Loss: {np.average(train_losses):.4f}, " +
              f"Acc: {np.average(train_corrects)*100:6.3f}%, LR: {get_lr(optimizer):.8f}")


""" Execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    """ Model """
    parser.add_argument("--node_number", type=int, default=NODE_NUMBER, help="Number of nodes")
    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH, help="Max number of nodes")
    parser.add_argument("--t_c", type=float, default=THRESHOLD_OF_CONNECTION, help="Threshold of connection")
    parser.add_argument("--max_n_c", type=int, default=MAX_NUMBER_OF_CONNECTION, help="Max number of connection")
    parser.add_argument("--min_n_c", type=int, default=MIN_NUMBER_OF_CONNECTION, help="Min number of connection")
    # parser.add_argument("--", action="store_true", help="")
    
    """ Training """
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of nodes")
    
    """ Misc """
    parser.add_argument("--output_path", type=str, help="Root directory path for output",
        default=f"history/{time.strftime('%Y.%m.%d', time.localtime())}")
        # default=f"history/{time.strftime('%Y.%m.%d/%H.%M.%s', time.localtime())}")
    
    opt = parser.parse_args()
    print(opt)
    main(opt)
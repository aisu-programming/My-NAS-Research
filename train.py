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
NODE_NUMBER = 64
MAX_DEPTH   = 4
THRESHOLD_OF_CONNECTION  = 0.75  # t_c
MAX_NUMBER_OF_CONNECTION = 4     # max_n_c >= 1
MIN_NUMBER_OF_CONNECTION = 1     # min_n_c >= 1


""" Function """
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def main(opt):

    """ Model """
    node_num, max_depth = opt.node_number, opt.max_depth
    threshold_of_connection = opt.t_c
    max_number_of_connection, min_number_of_connection = opt.max_n_c, opt.min_n_c
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    model = Model(node_num, max_depth, threshold_of_connection,
                  max_number_of_connection, min_number_of_connection, output_path)

    """ Data """
    batch_size = opt.batch_size
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True,  download=True, transform=transform)
    # train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [ 45000, 5000 ], torch_generator)
    test_dataset  = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    test_dataloader  = torch.utils.data.DataLoader( test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    """ Training """
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    """ Train """
    pbar = tqdm(enumerate(train_dataloader), ascii=True)

    for epoch in range(1, 100+1):

        model.search_path(plot_dag=True)

        losses = []
        correct_num = 0
        for bi, (batch_imgs, batch_labels) in pbar:

            batch_predictions = model(batch_imgs)
            loss = loss_fn(batch_predictions, batch_labels)
            loss.backward()
            optimizer.zero_grad()
            lr_scheduler.step(loss.item())
            
            correct_num += (batch_predictions==batch_labels).float().sum()
            accuracy = correct_num/max(bi*batch_size, len(train_dataloader))
            losses.append(loss)
            avg_loss = np.average(losses)
            pbar.set_description(f"Epoch {epoch:3}/100 [Train] Avg Loss: {avg_loss:.4f}, Acc: {accuracy*100:.3f}%, LR: {get_lr(optimizer):.8f}")


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
    parser.add_argument("--batch_size", type=int, default=NODE_NUMBER, help="Number of nodes")
    
    """ Misc """
    parser.add_argument("--output_path", type=str, help="Root directory path for output",
        default=f"history/{time.strftime('%Y.%m.%d', time.localtime())}")
    
    opt = parser.parse_args()
    print(opt)
    main(opt)
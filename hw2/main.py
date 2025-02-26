import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet18
from utils import Logger

logger: Logger = Logger()

def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ResNet for CIFAR-10')
    # logger settings
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--profile', action='store_true', default=False, help='enable profiling mode')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory for profiling (default: ./log)')
    # model settings
    parser.add_argument('--random_seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--batch_norm', action='store_true', default=True, help='use batch normalization')
    parser.add_argument('--compile', action='store_true', default=False, help='compile model with TorchInductor')
    # dataset settings
    parser.add_argument('--input_dir', type=str, default='./data', help='input directory for CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--worker', type=int, default=2, help='number of workers to load data (default: 2)')
    # training settings
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use (default: sgd)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    
    return parser

def fetch_dataloader(input_dir: str, batch_size: int, worker: int) -> DataLoader:
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(input_dir, train=True, download=True, transform=img_transform)
    logger.info(f'Loaded CIFAR-10 training dataset from "{input_dir}", Size: {len(train_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    logger.info(f'Created DataLoader with batch size {batch_size} and {worker} workers')
    return train_loader


def main(args: argparse.Namespace) -> None:
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    input_dir = args.input_dir
    batch_size = args.batch_size
    epochs = args.epochs
    worker = args.worker
    batch_norm = args.batch_norm
    optim = args.optim
    log_dir = args.log_dir

    torch.manual_seed(args.random_seed)

    resnet = ResNet18(batch_norm=batch_norm)
    logger.info(f'Created ResNet18 model [BatchNorm: {batch_norm}]')
    if optim == 'sgd':
        optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif optim == 'nestrov':
        optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(resnet.parameters(), lr=0.01, weight_decay=5e-4)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta(resnet.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=5e-4)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    else:
        raise ValueError(f'invalid optimizer: {optim}')
    
    logger.info(f'Using {optim.upper()} optimizer')

    train_loader = fetch_dataloader(input_dir, batch_size, worker)

    logger.info('Start training ResNet18 ...')
    logger.info(f'Using device: {device}')
    if args.profile:
        logger.info('Start profiling Mode ...')
        train_profile(resnet, epochs, train_loader, optimizer, device, log_dir)
    else:
        train(resnet, epochs, train_loader, optimizer, device)


def train(resnet: ResNet18, epochs: int, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> None:
    resnet = resnet.to(device)
    resnet.train()
    for e in range(epochs):
        epoch_elapsed = epoch_dataload = epoch_train = 0
        epoch_start = time.perf_counter()
        logger.debug(f"========= Epoch {e} =========")
        dataload_start = time.perf_counter()
        for i, batch in enumerate(dataloader):
            dataload_end = time.perf_counter()
            epoch_dataload += dataload_end - dataload_start

            img, label = batch
            img, label = img.to(device), label.to(device)
            
            train_start = time.perf_counter()
            output = resnet(img)
            loss = nn.CrossEntropyLoss()(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_end = time.perf_counter()
            epoch_train += train_end - train_start

            train_acc = (output.argmax(dim=1) == label).float().mean()
            logger.debug(f'batch {i}, loss: {loss.item(): .6f}, accuracy: {train_acc.item(): .4f}')

            dataload_start = time.perf_counter()
        epoch_end = time.perf_counter()
        epoch_elapsed = epoch_end - epoch_start
        logger.info(f'[Benchmark] - Epoch {e+1}, elapsed: {epoch_elapsed:.4f}s, dataload: {epoch_dataload:.4f}s, training: {epoch_train:.4f}s')


def train_profile(resnet: ResNet18, epochs: int, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str, log_dir: str) -> None:
    resnet = resnet.to(device)
    resnet.train()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,  # Monitor CPU
            torch.profiler.ProfilerActivity.CUDA  # Monitor GPU
        ],
        schedule=torch.profiler.schedule(
            wait=1,  # Skip first step for warmup
            warmup=1,  # Start recording after warmup
            active=3,  # Record profiling for 3 steps per epoch
            repeat=epochs  # Repeat schedule for multiple epochs
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{log_dir}/{time.time()}"),  # Save trace for TensorBoard
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for e in range(epochs):
            logger.debug(f"========= Epoch {e} =========")
            for i, batch in enumerate(dataloader):
                img, label = batch
                img, label = img.to(device), label.to(device)
                print(img.device)
                
                output = resnet(img)
                loss = nn.CrossEntropyLoss()(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_acc = (output.argmax(dim=1) == label).float().mean()
                logger.debug(f'batch {i}, loss: {loss.item(): .6f}, accuracy: {train_acc.item(): .4f}')

                prof.step()



if __name__ == '__main__':
    args = parser().parse_args()
    if args.debug:
        logger.set_level(10)
    main(args)
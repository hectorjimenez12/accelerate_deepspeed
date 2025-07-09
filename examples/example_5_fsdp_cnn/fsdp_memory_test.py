import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
from functools import partial
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint_sequential


# ------------------------- Model -------------------------
class DeepCNN(nn.Module):
    def __init__(self, width_multiplier=1, use_checkpoint=False):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(10):
            out_channels = 64 * width_multiplier
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, 10)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.features, 5, x)
        else:
            x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ------------------------- Setup -------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=180))
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets.MNIST('./data', train=True, download=True, transform=transform)
    datasets.MNIST('./data', train=False, download=True, transform=transform)


# ------------------------- Training -------------------------
def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2)

    model = DeepCNN(width_multiplier=args.width, use_checkpoint=args.checkpoint).to(device)
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e6)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler() if args.amp else None

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast(enabled=args.amp):
                output = model(data)
                loss = criterion(output, target)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**2
                reserved = torch.cuda.memory_reserved(device) / 1024**2
                print(f"[RANK {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Mem: alloc={allocated:.2f}MB, reserved={reserved:.2f}MB")

    if rank == 0:
        print("✅ Training complete")
    cleanup()


# ------------------------- CLI and Entry -------------------------
def run_fsdp(args):
    prepare_data()
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("⛔ Requires at least 2 GPUs for FSDP testing.")
        return
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size per GPU")
    parser.add_argument("--width", type=int, default=2, help="Width multiplier for DeepCNN")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    parser.add_argument("--checkpoint", action="store_true", help="Enable activation checkpointing")
    args = parser.parse_args()

    run_fsdp(args)

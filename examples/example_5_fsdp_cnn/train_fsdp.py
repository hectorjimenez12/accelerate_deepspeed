import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
import datetime


def check_gpu_availability():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    return num_gpus


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return self.fc(x)


def prepare_data():
    """Downloads MNIST data once before spawning distributed processes."""
    print("Downloading MNIST data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets.MNIST('./data', train=True, download=True, transform=transform)
    datasets.MNIST('./data', train=False, download=True, transform=transform)
    print("MNIST data download complete.")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120)
    )
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed(rank, world_size, epochs=2):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        try:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        except ImportError:
            def size_based_auto_wrap_policy(module, recurse, unwrapped_params, min_num_params=1e8):
                return sum(p.numel() for p in unwrapped_params) > min_num_params

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        fsdp_model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)

        for epoch in range(epochs):
            fsdp_model.train()
            sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = fsdp_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0 and rank == 0:
                    print(f"[RANK {rank}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            print("✅ Distributed training finished.")
    except Exception as e:
        print(f"❌ Error on rank {rank}: {str(e)}")
    finally:
        cleanup()


def train_single_gpu(epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

    print("✅ Single GPU training finished.")
    return model


def run_training():
    num_gpus = check_gpu_availability()

    if num_gpus >= 2:
        print(f"🚀 Running distributed training on {num_gpus} GPUs")
        world_size = num_gpus
        try:
            torch.multiprocessing.spawn(
                train_distributed,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            print(f"⚠️ Distributed training failed: {str(e)}. Falling back to single GPU.")
            train_single_gpu()
    else:
        print("🟡 Not enough GPUs for distributed training. Running single GPU training.")
        train_single_gpu()


if __name__ == "__main__":
    prepare_data()
    run_training()

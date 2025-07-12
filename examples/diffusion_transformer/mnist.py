from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import v2

import ornamentalist


@ornamentalist.configure()
def get_dataloaders(
    *,
    data_dir: str = ornamentalist.Configurable["./data"],
    batch_size: int = ornamentalist.Configurable[256],
    num_workers: int = ornamentalist.Configurable[8],
    pin_memory: bool = ornamentalist.Configurable[True],
) -> tuple[DataLoader, DataLoader]:
    train_dataset = MNIST(
        root=data_dir, train=True, download=True, transform=v2.ToImage()
    )
    val_dataset = MNIST(
        root=data_dir, train=False, download=True, transform=v2.ToImage()
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader

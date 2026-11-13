"""Train the ViT in model.py with PyTorch DistributedDataParallel (DDP).

Single node, 8 GPUs (one process per GPU, launched by torchrun):

    torchrun --standalone --nproc_per_node=8 train.py --data-dir ./data

Smoke test with random images (no dataset download needed):

    torchrun --standalone --nproc_per_node=8 train.py --synthetic --epochs 1

Resume from the last checkpoint:

    torchrun --standalone --nproc_per_node=8 train.py --resume checkpoints/last.pt

`--batch-size` is per GPU, so the global batch is batch_size * 8.
`--lr` is the peak learning rate for that global batch.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import ViT

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    p = argparse.ArgumentParser(description="ViT + DDP training on CIFAR-10")
    # Data
    p.add_argument("--data-dir", default="./data")
    p.add_argument(
        "--synthetic", action="store_true", help="use random images instead of CIFAR-10"
    )
    p.add_argument("--synthetic-size", type=int, default=8192)
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers per GPU")
    # Model
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    # Optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128, help="per-GPU batch size")
    p.add_argument(
        "--lr", type=float, default=1e-3, help="peak LR for the global batch"
    )
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 autocast (on by default; --no-amp to disable)",
    )
    p.add_argument(
        "--compile", action="store_true", help="wrap the DDP model in torch.compile"
    )
    # Bookkeeping
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--resume", default="", help="path to a checkpoint to resume from")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Distributed setup
# --------------------------------------------------------------------------- #
def setup_distributed():
    """Initialise the process group from the env vars torchrun sets."""
    # Allow plain `python train.py` as a 1-process run for debugging.
    if "RANK" not in os.environ:
        os.environ.update(
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # pin this process to one GPU
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:  # CPU fallback, for testing only
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size, device


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SyntheticImages(Dataset):
    """Deterministic random images; handy for smoke tests and throughput checks."""

    def __init__(self, n, num_classes=10, img_size=32, offset=0):
        self.n, self.num_classes, self.img_size = n, num_classes, img_size
        self.offset = offset                       # first seed; keeps splits disjoint
 
    def __len__(self):
        return self.n
 
    def __getitem__(self, i):
        if not 0 <= i < self.n:
            # Needed so plain iteration (`for x in ds`) stops: without __iter__, Python
            # calls __getitem__(0), (1), ... until IndexError is raised.
            raise IndexError(f"index {i} out of range for dataset of size {self.n}")
        idx = self.offset + i
        g = torch.Generator().manual_seed(idx)
        return torch.randn(3, self.img_size, self.img_size, generator=g), idx % self.num_classes


def build_datasets(args, local_rank):
    if args.synthetic:
        return SyntheticImages(args.synthetic_size), SyntheticImages(
            args.synthetic_size // 8
        )

    from torchvision import datasets, transforms

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # Only one process per node downloads; the rest wait so they don't race on the files.
    if local_rank == 0:
        datasets.CIFAR10(args.data_dir, train=True, download=True)
        datasets.CIFAR10(args.data_dir, train=False, download=True)
    dist.barrier()

    train_ds = datasets.CIFAR10(args.data_dir, train=True, transform=train_tf)
    val_ds = datasets.CIFAR10(args.data_dir, train=False, transform=val_tf)
    return train_ds, val_ds


def build_loaders(args, train_ds, val_ds):
    # DistributedSampler gives each rank a disjoint 1/world_size shard of the data.
    train_sampler = DistributedSampler(
        train_ds, shuffle=True, seed=args.seed, drop_last=True
    )
    # Note: if len(val_ds) isn't divisible by world_size, the sampler pads by repeating a few
    # samples, which very slightly skews metrics. CIFAR-10 (10k) splits evenly over 8 GPUs.
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    common = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, sampler=val_sampler, **common
    )
    return train_loader, val_loader, train_sampler


# --------------------------------------------------------------------------- #
# Optimiser and schedule
# --------------------------------------------------------------------------- #
def param_groups(model, weight_decay):
    """No weight decay on biases, LayerNorm weights, cls token and positional embedding."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(("cls_token", "pos_embed")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine(total_steps, warmup_steps, min_ratio):
    """LR multiplier: linear warmup, then cosine decay to `min_ratio`."""

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args):
    model.train()
    stats = torch.zeros(3, device=device)  # loss_sum, correct, count
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=args.amp
        ):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # DDP all-reduces (averages) grads here
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()  # per-step schedule

        with torch.no_grad():
            stats[0] += loss.detach() * y.size(0)
            stats[1] += (logits.argmax(1) == y).sum()
            stats[2] += y.size(0)

        if step % args.log_interval == 0:
            # rank 0's local loss; .item() syncs, so only do it at log steps
            print0(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}"
            )

    # Aggregate metrics over all ranks (every rank must make this call).
    dist.all_reduce(stats)
    elapsed = time.time() - t0
    return {
        "loss": (stats[0] / stats[2]).item(),
        "acc": (stats[1] / stats[2]).item(),
        "img_per_s": stats[2].item() / elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    stats = torch.zeros(3, device=device)  # loss_sum, correct, count
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=args.amp
        ):
            logits = model(x)
        stats[0] += F.cross_entropy(logits.float(), y, reduction="sum")
        stats[1] += (logits.argmax(1) == y).sum()
        stats[2] += y.size(0)
    dist.all_reduce(stats)
    return {"loss": (stats[0] / stats[2]).item(), "acc": (stats[1] / stats[2]).item()}


def save_checkpoint(path, raw_model, optimizer, scheduler, epoch, best_acc, args):
    torch.save(
        {
            "model": raw_model.state_dict(),  # unwrapped: no "module." prefix
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()

    # Different seed per rank -> different dropout masks / augmentations per GPU.
    # Model weights still start identical: DDP broadcasts rank 0's weights on wrap.
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.amp and device.type == "cpu":
        args.amp = False  # keep the CPU test path simple

    print0(
        f"world_size={world_size} device={device.type} "
        f"global_batch={args.batch_size * world_size}"
    )

    # Data
    train_ds, val_ds = build_datasets(args, local_rank)
    train_loader, val_loader, train_sampler = build_loaders(args, train_ds, val_ds)

    # Model
    raw_model = ViT(
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    print0(f"params: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M")

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])  # load before wrapping in DDP

    model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None)
    if args.compile:
        model = torch.compile(model)

    # Optimiser + schedule
    optimizer = torch.optim.AdamW(
        param_groups(raw_model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=device.type == "cuda",
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(
            total_steps=args.epochs * steps_per_epoch,
            warmup_steps=args.warmup_epochs * steps_per_epoch,
            min_ratio=args.min_lr / args.lr,
        ),
    )

    start_epoch, best_acc = 0, 0.0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        print0(f"resumed from {args.resume} at epoch {start_epoch}")
        del ckpt

    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)  # new shuffle each epoch, same across ranks
            tr = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch, args
            )
            va = evaluate(model, val_loader, device, args)

            is_best = va["acc"] > best_acc
            best_acc = max(best_acc, va["acc"])
            print0(
                f"[epoch {epoch}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f} (best {best_acc:.4f}) | "
                f"{tr['img_per_s']:.0f} img/s"
            )

            if rank == 0:  # only one process writes checkpoints
                last = os.path.join(args.ckpt_dir, "last.pt")
                save_checkpoint(
                    last, raw_model, optimizer, scheduler, epoch, best_acc, args
                )
                if is_best:
                    torch.save(
                        raw_model.state_dict(), os.path.join(args.ckpt_dir, "best.pt")
                    )
            dist.barrier()  # keep ranks in step around checkpointing

            if best_acc > 0.999:
                break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

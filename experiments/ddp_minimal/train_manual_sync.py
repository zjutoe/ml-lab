import os

import torch
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return rank, world_size, local_rank, device


def main():
    rank, world_size, local_rank, device = setup()

    # Every rank must start from the same model parameters.
    model = torch.nn.Linear(
        in_features=1,
        out_features=1,
        bias=False,
    ).to(device)

    with torch.no_grad():
        model.weight.fill_(1.0)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    # Different rank gets different local data.
    if rank == 0:
        x = torch.tensor([[1.0]], device=device)
        y = torch.tensor([[2.0]], device=device)
    else:
        x = torch.tensor([[2.0]], device=device)
        y = torch.tensor([[4.0]], device=device)

    # Forward.
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    # Local backward.
    optimizer.zero_grad()
    loss.backward()

    local_grad = model.weight.grad.item()

    print(
        f"rank={rank} "
        f"before sync: "
        f"loss={loss.item():.4f}, "
        f"grad={local_grad:.4f}, "
        f"weight={model.weight.item():.4f}"
    )

    # Manual gradient synchronization.
    for param in model.parameters():
        dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
        )
        param.grad /= world_size

    synced_grad = model.weight.grad.item()

    print(
        f"rank={rank} "
        f"after sync: "
        f"grad={synced_grad:.4f}"
    )

    # Same gradient -> same parameter update.
    optimizer.step()

    print(
        f"rank={rank} "
        f"after step: "
        f"weight={model.weight.item():.4f}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

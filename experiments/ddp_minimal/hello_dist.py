import os

import torch
import torch.distributed as dist


def main():
    # 1. Initialize the distributed process group.
    dist.init_process_group(backend="nccl")

    # 2. torchrun provides these values through environment variables.
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # 3. Bind this process to its local GPU.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(
        f"PID={os.getpid()} "
        f"rank={rank} "
        f"world_size={world_size} "
        f"local_rank={local_rank} "
        f"device={device}"
    )

    # 4. Each rank starts with a different value.
    x = torch.tensor([rank + 1.0], device=device)

    print(f"rank {rank}: before all_reduce, x={x.item()}")

    # 5. Sum x across every rank and return the result to every rank.
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"rank {rank}: after all_reduce, x={x.item()}")

    # 6. Clean shutdown.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

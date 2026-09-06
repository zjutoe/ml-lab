# Minimal Distributed Data Parallel Training

This experiment implements synchronous data-parallel training
in two ways:

1. Manual gradient synchronization using NCCL AllReduce.
2. PyTorch DistributedDataParallel (DDP).

## Execution Model

Each GPU is controlled by one process/rank.
Each rank holds a full model replica and processes a different
local minibatch.

After local backward passes, gradients differ across ranks.
They must be reduced before the optimizer update to keep model
replicas consistent.

## Manual Synchronization

Local gradients are synchronized with:

    all_reduce(gradient)
    gradient /= world_size

After synchronization, all ranks perform the same local
optimizer step and retain identical model parameters.

## DDP

DDP automatically performs gradient synchronization during
backward, removing the explicit AllReduce loop.

## Key Observation

Distributed data parallel training introduces communication
and synchronization overhead in addition to local computation.
This motivates the scaling experiments in the next stage.

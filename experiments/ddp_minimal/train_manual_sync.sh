#!/usr/bin/env bash

torchrun \
    --standalone \
    --nproc-per-node=2 \
    train_manual_sync.py

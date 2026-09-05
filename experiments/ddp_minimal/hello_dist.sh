#!/usr/bin/env bash

torchrun \
    --standalone \
    --nproc-per-node=2 \
    hello_dist.py

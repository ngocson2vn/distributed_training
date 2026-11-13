#!/bin/bash

export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=8 train.py --synthetic --epochs 100
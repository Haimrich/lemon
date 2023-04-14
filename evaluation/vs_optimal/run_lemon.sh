#!/bin/bash
mkdir -p results/lemon; 
( time lemon inputs/arch.yaml inputs/alexnet_conv1.yaml  -o results/lemon/; ) 2>&1 | tee results/lemon/logs.txt
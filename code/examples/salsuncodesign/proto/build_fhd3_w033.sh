#!/bin/bash
# Simplified estimators (input LayerNorm, tanh, asin(tanh) outputs), 3 residual
# blocks of width 1/3 (45 and 24 hidden units), 24 inputs per cycle, one
# output per cycle in the residual blocks and two in the output layer; the
# rotations of five columns at a time and the inverse DCT as two tasks
cd "$(dirname "$0")"
A="TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=3 BUILD_DIR=build/hw_w033"
E="-DEST_NH0=45 -DEST_NH1=24 -DEST_OPAR_O=2"
make xo $A EST_DEFS="$E" > build_hw_fhd_cu3_w033.log 2>&1 || { echo "make exit=$?" >> build_hw_fhd_cu3_w033.log; exit 1; }
make xclbin $A EST_DEFS="$E" >> build_hw_fhd_cu3_w033.log 2>&1
echo "make exit=$?" >> build_hw_fhd_cu3_w033.log

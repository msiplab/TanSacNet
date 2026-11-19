#!/bin/bash
# Simplified estimators (input LayerNorm, tanh, asin(tanh) outputs), 3 residual
# blocks of width 0.5 (68 and 36 hidden units), 24 inputs and one output per
# cycle (build/hw_w05), with the rotations of five columns at a time and the
# inverse DCT as two tasks
cd "$(dirname "$0")"
A="TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=3 BUILD_DIR=build/hw_rot"
E="-DEST_NH0=68 -DEST_NH1=36"
make xo $A EST_DEFS="$E" > build_hw_fhd_cu3_rot.log 2>&1 || { echo "make exit=$?" >> build_hw_fhd_cu3_rot.log; exit 1; }
make xclbin $A EST_DEFS="$E" >> build_hw_fhd_cu3_rot.log 2>&1
echo "make exit=$?" >> build_hw_fhd_cu3_rot.log

#!/bin/bash
# Simplified estimators (input LayerNorm, tanh, asin(tanh) outputs), 3 residual
# blocks, 24 inputs and one output per cycle (as build/hw_q10d)
cd "$(dirname "$0")"
A="TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=3 BUILD_DIR=build/hw_v2"
make xo $A > build_hw_fhd_cu3_v2.log 2>&1 || { echo "make exit=$?" >> build_hw_fhd_cu3_v2.log; exit 1; }
make xclbin $A >> build_hw_fhd_cu3_v2.log 2>&1
echo "make exit=$?" >> build_hw_fhd_cu3_v2.log

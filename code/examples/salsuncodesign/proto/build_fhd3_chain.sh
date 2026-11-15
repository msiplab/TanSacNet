#!/bin/bash
# FC with two outputs per cycle (sums as DSP cascades), LayerNorm in three tasks
cd "$(dirname "$0")"
A="TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=3 BUILD_DIR=build/hw_chain"
make xo $A > build_hw_fhd_cu3_chain.log 2>&1 || { echo "make exit=$?" >> build_hw_fhd_cu3_chain.log; exit 1; }
make xclbin $A >> build_hw_fhd_cu3_chain.log 2>&1
echo "make exit=$?" >> build_hw_fhd_cu3_chain.log

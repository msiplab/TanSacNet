#!/bin/bash
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh > /dev/null
SZY=${SZY:-40} SZX=${SZX:-40} BAND=${BAND:-10} HALO=${HALO:-0} vitis_hls -f run_cosim_band.tcl > cosim.log 2>&1
echo "exit=$?" >> cosim.log

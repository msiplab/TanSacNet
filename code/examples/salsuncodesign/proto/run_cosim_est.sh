#!/bin/bash
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh > /dev/null
vitis_hls -f run_cosim_est.tcl > cosim_est.log 2>&1
echo "exit=$?" >> cosim_est.log

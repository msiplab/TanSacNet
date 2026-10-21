#!/bin/bash
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh > /dev/null
vitis_hls -f run_hls.tcl > hls.log 2>&1
echo "exit=$?" >> hls.log

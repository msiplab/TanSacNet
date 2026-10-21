#!/bin/bash
# run_hls_band.sh <prj> <szy> <szx> <band> <halo>
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh > /dev/null
PRJ=$1 SZY=$2 SZX=$3 BAND=$4 HALO=$5 vitis_hls -f run_hls_band.tcl > $1.log 2>&1
echo "exit=$?" >> $1.log

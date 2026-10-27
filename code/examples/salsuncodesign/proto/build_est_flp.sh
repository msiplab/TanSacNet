#!/bin/bash
# est_kernel for hw (debugging the estimator on the card)
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh >/dev/null; source /opt/xilinx/xrt/setup.sh >/dev/null
export CPATH=/usr/include/x86_64-linux-gnu LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
P=xilinx_u250_gen3x16_xdma_4_1_202210_1; D=build/est_hw_flp
mkdir -p $D
( v++ -c -t hw --platform $P --save-temps --temp_dir $D/_x --hls.pre_tcl $PWD/flp.tcl -k est_kernel -I $PWD -o $D/est_kernel.xo est_kernel.cpp est_df.cpp && \
  v++ -l -t hw --platform $P --save-temps --temp_dir $D/_x --config est_kernel.cfg -o $D/est_kernel.xclbin $D/est_kernel.xo ) > build_est_flp.log 2>&1
echo "make exit=$?" >> build_est_flp.log

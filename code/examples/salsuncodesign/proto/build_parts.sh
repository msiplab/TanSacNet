#!/bin/bash
# est_parts.cpp kernels for hw (debugging the estimator on the card)
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2024.1/settings64.sh >/dev/null; source /opt/xilinx/xrt/setup.sh >/dev/null
export CPATH=/usr/include/x86_64-linux-gnu LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
P=xilinx_u250_gen3x16_xdma_4_1_202210_1; D=build/parts_hw
mkdir -p $D
( for k in part_ln part_fc part_gelu part_probe; do
    v++ -c -t hw --platform $P --save-temps --temp_dir $D/_x -k $k -I $PWD -o $D/$k.xo est_parts.cpp || exit 1
  done && \
  v++ -l -t hw --platform $P --save-temps --temp_dir $D/_x --config est_parts.cfg -o $D/est_parts.xclbin \
      $D/part_ln.xo $D/part_fc.xo $D/part_gelu.xo $D/part_probe.xo ) > build_parts.log 2>&1
echo "make exit=$?" >> build_parts.log

#!/bin/bash
cd "$(dirname "$0")"
make xclbin TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=1 BUILD_DIR=build/hw_cu1 > build_hw_fhd_cu1.log 2>&1
echo "make exit=$?" >> build_hw_fhd_cu1.log

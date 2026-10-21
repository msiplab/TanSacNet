# Build of the band kernel prototype (band_df.cpp, est_df.cpp) for Alveo U250
#
#   make xclbin TARGET=hw SZY=1080 SZX=1920 BAND=31 HALO=7 NCU=3
#   make xclbin TARGET=sw_emu SZY=40 SZX=40 BAND=6 HALO=7 NCU=1
#
# build/<TARGET>/salsun2d_band_df_<SZY>x<SZX>_b<BAND>h<HALO>_cu<NCU>.xclbin

TARGET   ?= sw_emu
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
SZY      ?= 40
SZX      ?= 40
BAND     ?= 6
HALO     ?= 7
NCU      ?= 1
KERNEL   := salsun2d_band_df
NAME     := $(KERNEL)_$(SZY)x$(SZX)_b$(BAND)h$(HALO)_cu$(NCU)
BUILD_DIR ?= build/$(TARGET)
XO       := $(BUILD_DIR)/$(NAME).xo
XCLBIN   := $(BUILD_DIR)/$(NAME).xclbin
LINK_CFG ?= band_df_$(NCU)cu.cfg

SHELL := /bin/bash
VPP   := source /tools/Xilinx/Vitis/2024.1/settings64.sh >/dev/null && source /opt/xilinx/xrt/setup.sh >/dev/null && v++
MULTIARCH := /usr/include/x86_64-linux-gnu
export CPATH        := $(MULTIARCH)$(if $(CPATH),:$(CPATH))
export LIBRARY_PATH := /usr/lib/x86_64-linux-gnu$(if $(LIBRARY_PATH),:$(LIBRARY_PATH))

VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) --save-temps \
             --temp_dir $(BUILD_DIR)/$(NAME)/_x --log_dir $(BUILD_DIR)/$(NAME)/logs \
             --report_dir $(BUILD_DIR)/$(NAME)/reports
DEFS := -DSZY=$(SZY) -DSZX=$(SZX) -DBAND=$(BAND) -DHALO=$(HALO)

.PHONY: xo xclbin emconfig
xo: $(XO)
xclbin: $(XCLBIN)

$(XO): band_df.cpp band_df.h est_df.cpp est_df.h gelu_table.h sin_table.h
	mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k $(KERNEL) -I $(CURDIR) $(DEFS) -o $@ band_df.cpp est_df.cpp

$(XCLBIN): $(XO) $(LINK_CFG)
	$(VPP) -l $(VPP_FLAGS) --config $(LINK_CFG) -o $@ $<

emconfig:
	source /tools/Xilinx/Vitis/2024.1/settings64.sh >/dev/null && emconfigutil --platform $(PLATFORM) --od $(BUILD_DIR)

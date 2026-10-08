# Build the SA-LSUN 2-D inference kernel for Alveo U250
#
# Usage:
#   make xclbin TARGET=sw_emu SZY=32 SZX=32   # software emulation (small frames)
#   make xo TARGET=hw                         # HLS synthesis only (reports)
#   make xclbin TARGET=hw                     # bitstream for the real card (hours)
#
# TARGET is sw_emu, hw_emu or hw. DESIGN is salsun2d_hls_opt (default),
# salsun2d_hls, or salsun2d_hls_band (streaming, one band of BAND block
# rows per kernel run; the kernel is hls/salsun2d_band_kernel.cpp). SZY x SZX
# is the frame size (default 300 x 300). The HDL Coder output goes to
# codegen/<SZY>x<SZX>/<DESIGN>/hdlsrc and the xclbin to
# build/<TARGET>/<DESIGN>_<SZY>x<SZX>[_b<BAND>].xclbin.
#
# Software emulation runs the kernel on one CPU thread, which takes many
# minutes per 300 x 300 frame; use small frames for it.
#
# Requires MATLAB R2026b with HDL Coder, Vitis 2024.1 and XRT. The
# settings scripts are sourced automatically.

TARGET   ?= sw_emu
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
KERNEL   := salsun2d_kernel
DESIGN   ?= salsun2d_hls_opt
SZY      ?= 300
SZX      ?= 300
BAND     ?= 31
ifeq ($(DESIGN),salsun2d_hls_band)
KERNEL   := salsun2d_band_kernel
NAME     := $(DESIGN)_$(SZY)x$(SZX)_b$(BAND)
CODEGEN_ARGS := [$(SZY) $(SZX)],'$(DESIGN)',BandRows=$(BAND)
else
NAME     := $(DESIGN)_$(SZY)x$(SZX)
CODEGEN_ARGS := [$(SZY) $(SZX)],'$(DESIGN)'
endif

VITIS_SETTINGS := /tools/Xilinx/Vitis/2024.1/settings64.sh
XRT_SETTINGS   := /opt/xilinx/xrt/setup.sh
SHELL := /bin/bash
VPP   := source $(VITIS_SETTINGS) >/dev/null && source $(XRT_SETTINGS) >/dev/null && v++

# The gcc bundled with Vitis does not search the Ubuntu multiarch directories
MULTIARCH := /usr/include/x86_64-linux-gnu
export CPATH        := $(MULTIARCH)$(if $(CPATH),:$(CPATH))
export LIBRARY_PATH := /usr/lib/x86_64-linux-gnu$(if $(LIBRARY_PATH),:$(LIBRARY_PATH))

ifeq ($(DESIGN),salsun2d_hls_band)
HLS_SRC_DIR := codegen/$(SZY)x$(SZX)_b$(BAND)/$(DESIGN)/hdlsrc
else
HLS_SRC_DIR := codegen/$(SZY)x$(SZX)/$(DESIGN)/hdlsrc
endif
HLS_CLASS   := $(HLS_SRC_DIR)/$(DESIGN)Class.hpp
BUILD_DIR   := build/$(TARGET)
XO          := $(BUILD_DIR)/$(NAME).xo
XCLBIN      := $(BUILD_DIR)/$(NAME).xclbin
# Linker configuration (e.g. LINK_CFG=hls/salsun2d_band_kernel_slr2.cfg)
LINK_CFG    ?= hls/$(KERNEL).cfg

ifeq ($(DESIGN),salsun2d_hls_opt)
DESIGN_FLAGS := -DSALSUN2D_OPT
endif
ifeq ($(DESIGN),salsun2d_hls_band)
DESIGN_FLAGS := -DBAND=$(BAND)
endif

VPP_FLAGS := -t $(TARGET) --platform $(PLATFORM) --save-temps \
             --temp_dir $(BUILD_DIR)/$(NAME)/_x --log_dir $(BUILD_DIR)/$(NAME)/logs \
             --report_dir $(BUILD_DIR)/$(NAME)/reports

.PHONY: all codegen xo xclbin emconfig clean cleanall

all: xclbin

codegen: $(HLS_CLASS)

$(HLS_CLASS): $(DESIGN).m salsun2d_hls_layout.m
	matlab -batch "run_salsun2d_hls_codegen($(CODEGEN_ARGS))"

xo: $(XO)

$(XO): hls/$(KERNEL).cpp $(HLS_CLASS)
	mkdir -p $(BUILD_DIR)
	$(VPP) -c $(VPP_FLAGS) -k $(KERNEL) -I $(HLS_SRC_DIR) \
	    -DSZY=$(SZY) -DSZX=$(SZX) $(DESIGN_FLAGS) -o $@ $<

xclbin: $(XCLBIN) emconfig

$(XCLBIN): $(XO) $(LINK_CFG)
	$(VPP) -l $(VPP_FLAGS) --config $(LINK_CFG) -o $@ $<

# emconfig.json is needed only for emulation targets
emconfig:
ifneq ($(TARGET),hw)
	source $(VITIS_SETTINGS) >/dev/null && emconfigutil --platform $(PLATFORM) --od $(BUILD_DIR)
endif

clean:
	rm -rf build _x .Xil *.log *.jou .run v++_*.log xcd.log

cleanall: clean
	rm -rf codegen

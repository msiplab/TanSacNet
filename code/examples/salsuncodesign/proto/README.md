# Dataflow angle estimator (prototype)

`est_df.cpp` is a hand-written Vitis HLS prototype of the angle estimator
of the band kernel as a layer pipeline: every layer (LayerNorm, FC, GELU,
FC, residual add, three times, then the output FC) is its own task, and
groups of NB = 15 blocks stream through them, so the throughput is that of
the slowest layer (an FC layer) instead of the sum of all layers. The
weights are loaded once into the memory of each FC task. The first
estimator only (135 features, 135 hidden units, 49 angles); the
LayerNorm is in fixed point.

Current form: one instance for the five estimators of a network with 3
residual blocks of width 1 (sizes and parameter offsets of estimator k at
run time, the parameters of all five in the memory of each task), NB = 5
lanes of P = 24 inputs (120 multiply-adds per cycle per FC task), the
weights in URAM, the inputs of the next group read during the current
one, the outputs sent during the last pass, and GELU by a table of 1025
points interpolated linearly (`gelu_table.h`).

C simulation (`vitis_hls -f run_csim.tcl`, test data from
`export_est_df.m`): the angles of all five estimators equal those of the
MATLAB fixed-point model (with the same GELU table) exactly.

Synthesis (`./run_hls.sh`, xcu250, 3.33 ns target): estimated clock
3.66 ns (FIFO accesses); DSP 1032, LUT 176k, FF 235k, BRAM 241, URAM 62;
810 cycles per group of 5 blocks for the FC tasks of the first estimator
(162 cycles per block, against 1487 in the current kernel), 216 for the
others (43, against 558).

## Band kernel (`band_df.cpp`)

A whole band kernel around the dataflow estimator: block DCT, five
analysis stages (each a dataflow of the feature gather, the estimator
engine and the rotations of a column), the coefficient mask (folded into
the last rotation), the synthesis rotations and atom extensions, and the
inverse DCT, as `salsun2d_hls_band.m` computes them. The coefficients of
the band are in two buffers of five banks (row mod 5, URAM); the angles
go to memory as 18-bit theta/pi (28 per 512-bit word) for the synthesis;
sin and cos come from an interpolated table (`sin_table.h`); the sign
parameters and the mask are applied as sign flips and zeros; the
rotations take one block per cycle, rotation-major over the blocks of a
column; the LayerNorm normalization is in fixed point.

C simulation (`run_csim_band.tcl`, data from `export_band_df.m`), 40 x 40
frames, one band without halo or two 6-row bands with halo 7: max error
against the MATLAB fixed-point model 1.3e-5 (max |y| 0.155).

Synthesis for Full HD (1080 x 1920, 31-row bands, halo 7): estimated
clock 3.77 ns; LUT 267k (62% of an SLR), FF 361k, DSP 1250, BRAM 353,
URAM 222 (69%). About 11.4 M cycles per band (estimators 7.2 M,
synthesis rotations 2.9 M), 34 M per frame for each of three compute
units (three bands each): about 0.11-0.13 s/frame at 260-300 MHz.

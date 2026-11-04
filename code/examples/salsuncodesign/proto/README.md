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
go to memory as 18-bit theta/pi for the synthesis, an angle of the five
blocks of a group as a row, five rows to a 512-bit word (no transpose);
sin and cos come from an interpolated table (`sin_table.h`); the sign
parameters and the mask are applied as sign flips and zeros; the
rotations take one block per cycle, rotation-major over the blocks of a
column; the LayerNorm normalization is in fixed point. The feature
gather gives one row of the five lanes per cycle over the nine
neighbors (the blocks of the next neighbor read meanwhile), and the
rotation takes the angles one row per cycle.

C simulation (`run_csim_band.tcl`, data from `export_band_df.m`), 40 x 40
frames, one band without halo or two 6-row bands with halo 7: max error
against the MATLAB fixed-point model 1.3e-5 (max |y| 0.155).

Synthesis for Full HD (1080 x 1920, 31-row bands, halo 7): estimated
clock 3.66 ns; LUT 259k, FF 280k, DSP 1258, BRAM 370, URAM 190. Per
column of a band, the gather takes 1299-1866 cycles and the rotation
879-2067 (4538-5105 and 5038-6928 before), below the estimator from
stage 1 on (about 2300); stage 0 is bound by its first FC (about 7650).

On the card (Alveo U250, three compute units at 300 MHz): 0.138 s/frame
with a frame per unit (0.160 before, 0.199 before that), 0.131 s/frame
over 60 s of calls, 0.149 with a frame split over the units; the output
equals the fixed-point model (max error 4.4e-5). 12.6 M cycles per band
for a unit (15.4 M before), against about 12.7 M from the synthesis
estimates. Two changes to the reading of the input: `src` has an m_axi
bundle of its own (sharing one with `dst`, whose writes are 128 bits
wide, the DCT read the float pixels one per beat in bursts of 16 with
one burst in flight, about 2 M cycles per band), and the DCT requests
the runs of the next column ahead (`hls::burst_maxi`) and reads a column
while it transforms the previous one (about 0.5 M cycles less). The
link (`band_df_3cu.cfg`) runs `phys_opt_design` after placement: without
it the write enables of the URAM buffers A and B, which have no input
register, failed timing (286.5 MHz). Board power 61.9 W under load,
50.5 W idle: 8.1 J/frame (9.6 before, 11.3 before that).

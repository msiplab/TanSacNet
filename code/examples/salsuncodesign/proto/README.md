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

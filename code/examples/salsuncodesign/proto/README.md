# Dataflow angle estimator (prototype)

`est_df.cpp` is a hand-written Vitis HLS prototype of the angle estimator
of the band kernel as a layer pipeline: every layer (LayerNorm, FC, GELU,
FC, residual add, three times, then the output FC) is its own task, and
groups of NB = 15 blocks stream through them, so the throughput is that of
the slowest layer (an FC layer) instead of the sum of all layers. The
weights are loaded once into the memory of each FC task. The first
estimator only (135 features, 135 hidden units, 49 angles); the
LayerNorm is in fixed point.

Synthesis (`./run_hls.sh`, xcu250, 3.33 ns target): estimated clock
2.433 ns; DSP 1689, LUT 195k, FF 295k, BRAM 636, URAM 24; 2877 cycles per
group of 15 blocks for each FC task (192 cycles per block, against 1487
for the first estimator of the current kernel), GELU 800, LayerNorm 360.
Not yet simulated against the MATLAB model.

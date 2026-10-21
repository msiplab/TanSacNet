# SA-LSUN 2-D: GPU training and FPGA inference

HW/SW co-design of the 2-D SA-LSUN in `../salsun/main_salsun2d.m`: the
network parameters are trained on the GPU, and inference (analysis, coefficient
mask and synthesis) runs on an Alveo U250.

```
temsip07 (GPU)                  temsip02 (FPGA build)                temsip07 (FPGA)
train on GPU ──> params (.mat) ───────────────────────────────────> FPGA inference ──> compare with GPU
                                 MATLAB reference -> HDL Coder ->
                                 Vitis HLS -> v++ -> xclbin ───────> (copy xclbin)
```

## Status

| Step | Description | Status |
|---|---|---|
| 1 | MATLAB reference implementation without Deep Learning Toolbox | done |
| 2 | HLS-friendly rewrite (fixed sizes, loops, parameters as arguments) | done |
| 3 | HDL Coder -> Vitis HLS, `sw_emu`, hardware build on temsip02 | whole-frame builds failed in routing (see Slack); streaming band design built (237.7 MHz) and validated on the U250: 10 frames match the reference to 8e-7; 90-lane engine failed timing (congestion); fixed-point estimators validated on the U250 (SLR3, 234.0 MHz, 0.43 s/frame); retrained estimators with 3 residual blocks of width 1: 248.5 MHz, 0.29 s/frame (slower than an optimized CPU or GPU implementation; see "Fair comparison with CPUs and GPUs") |
| 4 | GPU training -> FPGA inference -> comparison in one script on temsip07 | |
| 5 | Fixed-point conversion and accuracy evaluation | fixed-point estimators in the band design, validated on the U250 |

## Files

| File | Description |
|---|---|
| `salsun2d_extract_params.m` | Extracts all parameters from a `dlnetwork` (Mode `'Whole'`, ThetaMode `'Reuse'`, optional `Lv1_AcMask`) into a plain struct |
| `salsun2d_infer.m` | Reference inference: `[y,coefs,thetas] = salsun2d_infer(x,params)` |
| `salsun2d_cast_params.m` | Casts the parameters, e.g. to double for a high-precision reference |
| `Salsun2dInferTestCase.m` | Compares `salsun2d_infer` with `predict` of the `dlnetwork` |
| `salsun2d_base_field.m` | Base field separation of a sequence: none, batch time average, or causal first-order IIR (`Rho`), on the whole frame or the block-DC part only |
| `salsun2d_infer_sequence.m` | Frame-by-frame inference of a sequence with base field and causal standardization statistics (`'image'`, `'previous'`, `'ema'`, `'fir2'`); reference for a streaming implementation |
| `Salsun2dSequenceTestCase.m` | Tests of the two above and of the `Statistics` option of `salsun2d_infer` |
| `salsun2d_wave_data.m` | Wave equation data of `../salsun/createdata_waveEq.m`, vectorized |
| `measure_receptive_field.m`, `measure_statistics_stability.m` | Measurements for the discussion on tiled/streamed processing (receptive field, frame-to-frame stability of the standardization statistics) |
| `salsun2d_hls_band.m` | Streaming HLS design: one band of `BandRows` block rows with `L.Halo` = 7 rows of circular context; angles on chip; statistics given per channel; returns the band and the channel sums |
| `hls/salsun2d_band_kernel.cpp`, `.cfg` | Vitis kernel: one frame per run, bands with halo loaded from DDR, valid rows stored, sums accumulated; placed in SLR0 |
| `mex/salsun2d_band_mex.cpp`, `salsun2d_band_u250.m` | Run the band kernel from MATLAB (one frame per call; statistics policy on the host) |
| `salsun2d_band_frame.m`, `salsun2d_band_sequence.m`, `salsun2d_stats_from_sums.m` | Host logic of the band kernel in MATLAB (same band schedule), sequence driver with previous / EMA / FIR statistics |
| `Salsun2dHlsBandTestCase.m` | Band design vs. reference stream, frame assembly, moved last band |
| `salsun2d_band_fixed_model.m`, `salsun2d_round_fixed.m` | Reference model of the fixed-point estimators of the band design (rounded weights, quantizer), and the rounding of `ap_fixed` (nearest, saturate) |
| `salsun2d_calibrate_formats.m`, `salsun2d_static_quantizer.m`, `evaluate_salsun2d_fixed_estimator.m` | Fixed-point formats from calibration frames, quantizer with fixed formats, and the evaluation of the estimators with fixed formats |
| `salsun2d_check_band_mask.m` | Checks that the coefficient mask suits the band design (last stage reduced to 7 angles) |
| `salsun2d_infer_klt.m` | SA-LSUN with local KLT (local block PCA) rotations instead of the estimators, for the study below |
| `salsun2d_hls_layout.m` | Fixed configuration of the HLS version and offsets of each parameter in the packed vector |
| `salsun2d_pack_params.m` | Packs the parameters into one single vector (489,085 values) and checks the configuration |
| `salsun2d_hls.m` | HLS version: `y = salsun2d_hls(x,w)`, explicit loops only |
| `Salsun2dHlsTestCase.m` | Compares `salsun2d_hls` with `salsun2d_infer` |
| `salsun2d_create_test_network.m` | Network with randomly perturbed parameters, used by the tests |
| `salsun2d_hls_tb.m`, `run_salsun2d_hls_codegen.m` | HDL Coder (Vitis HLS 2024.1) code generation into `codegen/salsun2d_hls/hdlsrc` |

```matlab
params = salsun2d_extract_params(reconnet);   % reconnet as in main_salsun2d.m
y = salsun2d_infer(single(x),params);         % same as reconnet.predict
w = salsun2d_pack_params(params);
y = salsun2d_hls(single(x),w);                % HLS version, same result
runtests({'Salsun2dInferTestCase','Salsun2dHlsTestCase'})
run_salsun2d_hls_codegen([300 300])           % Vitis HLS C++ for 300 x 300 images
```

## Streaming options (reference only)

`salsun2d_infer_sequence` adds two options that a streaming FPGA
implementation can realize causally, without waiting for the whole frame
or sequence. Both are open loop (computed from the original frames):

- `BaseField='iir'`: a leaky integrator b_t = Rho*b_(t-1) + (1-Rho)*base(u_t)
  of the block-DC part (`Scope='dc'`) or the whole frame; the network
  processes u - b_(t-1) and b_(t-1) is added back to the output.
- `Statistics='previous'|'ema'|'fir2'`: the estimator inputs are
  standardized with the statistics of the previous frame(s) instead of
  the current frame. `salsun2d_infer(x,params,Statistics=stats)` takes
  such statistics and returns the measured ones.

With `BaseField='none'` and `Statistics='image'` the result equals
`salsun2d_infer` frame by frame; with a base field and no mask the
reconstruction is still perfect. The HLS design does not have these
options yet; they are meant for evaluation with trained networks first.

## Structure of the inference

Analysis: block DCT -> initial rotation -> [atom extension -> intermediate
rotation] x 4 (right, left, down, up) -> coefficient mask. Synthesis runs the
same stages in reverse with transposed rotations and the final rotation, then
the block IDCT. Only the analysis side has angle estimators (5 of them); the
synthesis side reuses their angles.

Each estimator: neighbor blocks (3 x 3, circular boundary) -> standardization
over all blocks of the image -> 3 residual blocks (LayerNorm, FC, GELU with
tanh, FC, skip) -> FC to the angles. With block 4 x 4, overlap 3 x 3 and
width 2, this is about 480k multiply-accumulates per block, almost all in the
estimators.

## Streaming design for the FPGA (band-wise, overlap-save)

`salsun2d_hls_band` implements the streaming form validated with
`salsun2d_infer_stream`: a frame is processed in bands of `BAND` block
rows, each with `L.Halo` = 7 block rows of circular context above and
below (6 for the structural receptive field of the analysis, 1 for the
synthesis; with this halo a band equals whole-frame processing exactly).
The angles of a band stay on chip, the standardization statistics are an
input (per channel and estimator) and the channel sums of the valid
blocks are an output, so the policy (previous frame, EMA, 2-tap FIR)
stays on the host.

```sh
make xclbin TARGET=sw_emu DESIGN=salsun2d_hls_band SZY=32 SZX=32 BAND=2
make xclbin TARGET=hw DESIGN=salsun2d_hls_band BAND=16        # 300 x 300, 5 bands
```

```matlab
build_mex_salsun2d
y = salsun2d_band_u250(u,params,'hw',BandRows=16,Statistics='fir2');
```

Software emulation (32 x 32, band 2, trained parameters): kernel vs.
whole-frame reference 5e-7, statistics within 1e-6. HLS for 300 x 300
with 16-row bands: URAM 38%, BRAM 24%, DSP 14%, LUT 42% of one SLR,
Fmax 369 MHz (two floats per URAM word with `ARRAY_RESHAPE`; without it
URAM is 86%). Compute overhead (16 + 14)/16 = 1.9x, plus the moved last
band: 5 bands of 30 rows for 75 rows, 2.0x.

Hardware (U250, temsip02, 2024.1): the build runs through place and
route without congestion warnings (3 h 46 m); the kernel clock is scaled
to 237.7 MHz. Ten frames of the wave data (300 x 300, FIR-2 statistics)
match the whole-frame reference with the same statistics to 8e-7, and
the statistics are exact because the rows repeated by the moved last
band are left out of the sums (`skipRows`). Throughput: 2.7 s/frame on
the card against 1.06 s/frame for the MATLAB reference on the CPU; the
kernel is a single unpipelined band engine (15 lanes), so the remaining
work is throughput, not correctness.

### Throughput of the band engine

Cycle budget of the 15-lane engine from the HLS report, per band of 16
rows (about 127 M cycles measured): fully connected layers 62% (two
groups of 15 blocks per weight), rotations 12-18% (one Givens generator
per block, sequential products), LayerNorm / GELU / residual add 15%,
feature extraction 4%, band load and store under 1% (so overlapping the
DDR transfers with the computation would gain nothing). The revised
engine processes a group of `L.NColGroup` = 3 columns at a time with one
lane per block in the fully connected layers (90 lanes, one weight per
cycle), pipelines the feature extraction, the residual add and the
LayerNorm (15 lanes), the GELU (3 tanh units) and the rotation products,
and shares one instance of each rotation function between analysis and
synthesis. HLS estimate: about 39 M cycles per band (3.3x), LUT 46%,
DSP 33%, BRAM 34%, URAM 19% of one SLR. Two pitfalls met on the way:
`mod` on doubles inside a pipelined loop becomes a 2100-cycle `fmod`
replicated per pipeline stage (1.2 M LUT), so circular indices are
integer; and the two generator calls of an unrolled block pair are
serialized by HLS on one instance.

The 90-lane engine routed but failed timing: congestion level 6-7 in
SLR0, the worst kernel path (in the 15-lane LayerNorm, reaching the 90
banks of the intermediate arrays) had 6.3 ns of route delay, and the
DDR clocks of the shell missed their target, which stops the build. The
engine is now built with one column group (`L.NColGroup` = 1, 30 lanes
in the fully connected layers: one weight per cycle for the 30 blocks of
a column, 2x the 15-lane engine), LUT 34%, DSP 23%, BRAM 12% of one
SLR. `hls/salsun2d_band_kernel_slr2.cfg` (with `LINK_CFG=`) places the
kernel in SLR2 with DDR[2] and congestion-oriented directives, for
trying the 90-lane engine away from the shell.

Hardware, 30-lane engine with the matrix rotations (one column group):
built in 3 h 27 m with congestion warnings but without failing shell
clocks, kernel clock 238.8 MHz; ten frames match the reference to
7.5e-7; 1.52 s/frame for the kernel call (1.61 s/frame with the host
policy), against 2.68 s/frame for the 15-lane engine (1.76x) and 0.92
s/frame for the MATLAB reference on the CPU.

### The estimators predict rotation angles

Measured with the trained parameters (frames 11-15, MSE relative to the
reference): angle noise of 0.003 / 0.01 / 0.03 rad gives 1.03 / 1.28 /
3.5, so the angles need about 11-12 bits; constant angles (no
adaptation) give 162 and angles computed on every other block 86, since
neighboring angles differ by 0.11-0.35 rad against a spread of 0.3-0.9
rad (the local adaptation is essential). The output layer is 3% of the
multiply-adds of the estimators; the residual weights are nearly full
rank (rank 125 of 135 for 99% of the energy), so a low-rank
approximation without retraining does not work (half rank: 3.5).

Two consequences are used in the band design without changing the
model:

* With the mask keeping channels 1 and 9, only the first row of the
  last rotation matters, which depends only on its first Pa-1 = 7
  angles (pairs (1,2)..(1,8)); noise of 1 rad on the other 21 angles
  leaves the MSE unchanged. The last stage applies 7 rotations, and the
  last estimator stores 7 angles (`L.LastStageNAngles`,
  `salsun2d_check_band_mask`).
* The rotations are applied to the coefficients directly as the
  sequence of Givens rotations (synthesis: transposes in reverse order)
  instead of forming the matrix and multiplying, with the cosines and
  sines of a column computed first. HLS: 0.14-0.18 M cycles per call
  with a fixed latency, against 0.8-5.3 M before (about 1.7 M instead
  of 10-33 M cycles per band); LUT 30%, DSP 20% of one SLR.

Hardware, 30-lane engine with the direct Givens rotations: built in 4 h
20 m, kernel clock 262.9 MHz; ten frames match the reference to 6.9e-7;
1.01 s/frame for the kernel call (1.12 s/frame with the host policy),
1.5x the matrix rotations and 2.65x the 15-lane engine, against 0.94
s/frame for the MATLAB reference on the CPU. About 53 M cycles per band,
of which the fully connected layers are about 36 M. The 90-lane engine
placed in SLR2 (`salsun2d_band_kernel_slr2.cfg`) routed but missed a
DDR clock of the shell by 0.004 ns.

## HLS version (whole frame)

`salsun2d_hls` fixes the configuration of `main_salsun2d.m` (stride [4 4],
overlapping factor [3 3], no DC leakage, neighbor [3 3], 3 residual blocks,
width 2) and takes all parameters in one vector `w`, so that retraining on
the GPU does not require rebuilding the FPGA kernel. Two observations keep it
simple:

- The features of an estimator are circular shifts of the same channels, so
  their mean and variance over the image equal those of the channels. The
  standardization statistics are computed once per channel.
- Each estimator reads the neighbors of a block before rotation, so rotated
  coefficients are written to a second buffer.

HDL Coder generates Vitis HLS C++ for 300 x 300 images with no conformance
errors or warnings. The generated C++, compiled with the Vitis HLS math
library (C simulation, 32 x 32 with mask), differs from the double-precision
reference by 7.7e-4, against 6.7e-4 for the single-precision MATLAB version.
The generated code is not optimized yet (everything is inlined into one
function, no pragmas); that is step 3.

## Fixed-point estimators of the band design

`evaluate_salsun2d_fixed_estimator` rounds the estimator signals with
formats fixed in advance (`salsun2d_calibrate_formats` on frames 1-10,
`salsun2d_static_quantizer`), as hardware needs, and reports the MSE on
frames 11-15 relative to the floating-point network. With one format per
signal and estimator, 14-bit signals and 12-bit weights cost 0.1-0.2%.
With `Common=true`, one format for all inputs of the fully connected
layers (features, LayerNorm and GELU outputs of every estimator) and one
for all weight matrices, so that one fixed-point layer serves every
call:

| signals | weights | MSE / float |
|---|---|---|
| 14 bits | 14 bits | 1.0011 |
| 16 bits | 16 bits | 1.0002 |
| 16 bits | 18 bits | 1.0001 |
| 18 bits | 18 bits | 1.0001 |

The band design uses 18-bit signals (11 fraction bits, one bit of
headroom), 18-bit weights (16 fraction bits) and 48-bit accumulators
(`L.FixSignal`, `L.FixWeight`, `L.FixAcc`), the widths of the DSP
multipliers and block RAMs. The fully connected layers multiply-add
`L.FcInputs` = 4 inputs per lane and cycle (120 multipliers for 30
lanes, one DSP each) with exact accumulation; the weights are packed
row by row (`salsun2d_pack_params(...,RowMajorWeights=true)`) and read
as two aligned words of four. The LayerNorm sums are exact fixed-point
sums (II = 1); the standardization, the LayerNorm scaling and the GELU
stay in single precision and are rounded to the signal format; the data
path (DCT, rotations, atom extensions) stays in single precision.

In MATLAB the band design emulates the fixed-point values in double
precision with the same rounding (`salsun2d_round_fixed`), so its tests
stay fast; code generation produces `ap_fixed` types. The reference is
`salsun2d_infer` with `salsun2d_band_fixed_model` (rounded weights and a
quantizer for the three signal classes, `EstimatorTags=true`). The
code generation turns off integer saturation
(`SaturateOnIntegerOverflow = false`): the saturating casts hid from HLS
that the four weights lie in two words (II 2-4 instead of 1).

Results (MATLAB emulation and `sw_emu`): band design vs. fixed-point
model 2.0-2.5e-5, fixed-point model vs. float 4.6e-5 (max |y| 0.2). HLS:
estimator call of the first stage at most 12.2 M cycles against 43.8 M
in single precision; LUT 44%, DSP 20%, BRAM 17%, URAM 19% of one SLR.

Hardware: in SLR0 the build failed routing (conflicts in the DDR
controller of the shell) and in SLR3 it first missed a shell clock by
7 ps; with the post-route physical optimization enabled
(`hls/salsun2d_band_kernel_slr3.cfg`, DDR[3]) it builds in 4 h 57 m at
234.0 MHz. Ten frames (300 x 300, FIR-2 statistics) match the
fixed-point model to 3e-5 and the floating-point network to 2.1e-4 with
the same MSE (3.633e-6 on frame 1 for both). The kernel call takes 0.43
s/frame (0.55 s/frame with the host policy), 2.3x faster than the
direct-Givens single-precision engine and 2.3x faster than the MATLAB
reference on the CPU (0.97 s/frame); about 20 M cycles per band.

## Band design with smaller estimators

`make xclbin ... RES=3 WIDTH=1` builds the band design for retrained
estimators with 3 residual blocks of width 1 (`salsun2d_band_config`,
`salsun2d_estimator_size`; the host picks the matching xclbin). Two
changes were needed: the intermediate arrays of the estimators hold
max(features, hidden) rows (with width 0.5 they were too small, which
only the generated code showed), and the rotations are applied rotation
by rotation over the blocks of a column (II = 1) instead of unrolling the
28 rotations of a block (II = 28), whose wiring a congestion analysis of
a failed build showed in 75-96% of the congested windows.

Hardware (U250, SLR3, `hls/salsun2d_band_kernel_slr3.cfg`), parameters
trained for 800 epochs with distillation: 248.5 MHz; ten frames match the
fixed-point model to 2.7e-5 with the MSE of the floating-point network;
0.29 s/frame for the kernel call (0.41 s/frame with the host policy),
against 0.43 s/frame with the current estimators and 0.95 s/frame for
the MATLAB reference on the CPU (an unoptimized reference; see below).
The design with 4 residual blocks of width 0.5 failed routing (conflicts
inside the kernel).

## Fair comparison with CPUs and GPUs

The CPU times above are those of `salsun2d_infer`, a block-by-block
reference that favors the FPGA. `salsun2d_infer_fast` computes the same
frame with every step vectorized over all blocks (matrix products for the
fully connected layers, Givens rotations on the coefficient rows of all
blocks at once), in single precision with the same last-stage shortcut as
the kernel. It matches the reference to 3e-7 and runs in MATLAB, on
`gpuArray` inputs, and as generated C (MATLAB Coder MEX). Seconds per
300x300 frame (median of repeated calls, data already on the device):

| Implementation | R3 W2 | R3 W1 |
|---|---|---|
| CPU, MATLAB, 1 thread (temsip02) | 1.22 | 0.64 |
| CPU, MATLAB, 36 threads (temsip02) | 0.144 | **0.113** |
| CPU, generated C (MEX), 1 thread | 0.88 | 0.50 |
| GPU A800 (temsip01) | 0.051 | **0.049** |
| GPU RTX A6000 (temsip08) | 0.138 | 0.139 |
| GPU Quadro RTX 6000 (temsip07) | 0.157 | 0.157 |
| FPGA U250, one compute unit (kernel call) | 0.43 (current estimators) | 0.29 |

With one compute unit (0.29 s/frame) the FPGA kernel was about 2.6 times
slower than the multithreaded CPU and 6 times slower than the A800; the
three-unit kernel with whole frames (0.040 s/frame, below) is 2.7 times
faster than the multithreaded CPU and 1.2 times faster than the A800. The generated C does not use more threads
(no `parfor`, matrix products without an optimized BLAS) and is not the
best CPU figure. The FPGA is a single 300 MHz-class compute unit using a
fraction of the device; more compute units, wider FC lanes, or
energy per frame are where it can compete.

Energy per frame (R3 W1, 60 s of frames back to back, the same frames as
above). Board power from `xrt-smi examine -r electrical` (U250) and
`nvidia-smi` (GPUs); CPU package plus DRAM power from RAPL (both sockets
of temsip02, Xeon Gold 6240 x 2). The FPGA and GPU figures leave out the
host CPU that drives them; "above idle" subtracts the power of the idle
device (U250 with the xclbin loaded):

| Device | s/frame | Idle W | Load W | J/frame | J/frame above idle |
|---|---|---|---|---|---|
| CPU, 36 threads | 0.109 | 57.0 | 247.1 | 26.9 | 20.7 |
| CPU, 1 thread | 0.555 | 57.0 | 115.2 | 63.9 | 32.3 |
| GPU A800 | 0.047 | 44.6 | 97.9 | **4.6** | 2.5 |
| GPU RTX A6000 | 0.139 | 78.9 | 90.1 | 12.5 | 1.6 |
| GPU Quadro RTX 6000 | 0.157 | 51.4 | 56.9 | 9.0 | 0.87 |
| FPGA U250, one compute unit | 0.288 | 31.4 | 32.9 | 9.5 | **0.45** |
| FPGA U250, two compute units (packed angles) | 0.178 | 36.3 | 38.9 | 6.9 | 0.46 |
| FPGA U250, two compute units, 46-row bands | 0.103 | 36.9 | 40.2 | 4.1 | 0.34 |
| FPGA U250, two compute units, whole frames | 0.062 | 37.8 | 41.2 | 2.6 | 0.21 |
| FPGA U250, three compute units, whole frames | **0.040** | 45.3 | 51.3 | **2.1** | 0.24 |

The FPGA uses a third of the energy of the multithreaded CPU per frame,
about as much as the Quadro RTX 6000 and twice as much as the A800. Its
9.5 J are almost all the idle power of the card (31 W; the kernel adds
1.5 W), so more frames per second from the same card lower the energy per
frame almost in proportion.

Two compute units (`hls/salsun2d_band_kernel_2cu.cfg`, SLR3 with DDR[3]
and SLR2 with DDR[2]) failed routing with single-precision angles in
every placement (SLR3 + SLR2, SLR3 + SLR1, SLR3 + SLR0); the angle store
(46 URAM) and its multiplexers appeared in all congested regions. With
the angles stored as 18-bit fractions of pi, four per URAM word (22
URAM; `L.FixAngle`, `salsun2d_round_angle`), the SLR3 + SLR2 build
routed at 235.4 MHz: ten frames match the fixed-point model to 2.7e-5,
0.178 s/frame for the kernel call (0.32 s/frame with the host policy),
6.9 J/frame. A one-unit build of the same kernel in SLR3 failed routing
(20977 node overlaps), so routing in one SLR remains marginal.

Further changes of the two-unit kernel: the DCT, inverse DCT, mask and
channel-sum loops at II = 1 (they had II = 4 to 11), eight FC inputs per
lane from weight rows aligned to words of 8
(`salsun2d_hls_layout(...,AlignedWeights=true)`), and the estimators
processing the blocks of a column in groups of up to 15 rows
(`L.NFcLanes`) instead of one lane per block row. The last change is
what made the builds route again (the FC lanes and their memory banks
were in the congested windows of the failed builds), and it frees the
band height from the lanes. With 46-row bands (60 rows with halo, two
bands per frame, one per compute unit) the kernel computes 120 block
rows per frame instead of 150 and built at 270.4 MHz: ten frames match
the fixed-point model to 2.4e-5, 0.103 s/frame for the kernel call
(0.24 s/frame with the host policy), 4.1 J/frame, less than the A800.
The same kernel with 16-row bands: 237.9 MHz, 0.17 s/frame.

Whole frames: a band of the whole frame height needs no halo (the
circular boundary of the frame is that of the band; Makefile BAND=75
HALO=0), 75 block rows per frame. Each compute unit takes one whole
frame, two consecutive frames at a time (`salsun2d_band_mex` with
szy x szx x 2 inputs, `salsun2d_band_u250(...,Halo=0,FramesPerCall=2)`),
both with the statistics of the two frames before them (the MSE grows by
0.3-0.7% on the wave simulations for the one frame more of delay of the
second frame). Built at 284.4 MHz: ten frames match the fixed-point
model with the same statistics to 3.2e-5, 0.124 s per call of two
frames, 0.062 s/frame (0.066 s/frame with the host for 100 frames,
`InitFast=true`), 2.6 J/frame.

The host adds little in the steady state: the host-inclusive times of
ten frames were dominated by the statistics of the first frame from the
fixed-point reference (about 1.5 s); `InitFast=true` measures them with
`salsun2d_infer_fast` (0.15 s).

Three compute units (`hls/salsun2d_band_kernel_3cu.cfg`: SLR3, SLR2 and
SLR0 with their DDR banks; SLR1 holds most of the shell), each with one
whole frame of a group of three (`FramesPerCall=3`; the MSE grows by
0.8-1.4% on the wave simulations), with the FC layers of the estimators
in one pipeline over their input groups (the inputs of the next group
read during the current one): built at 261.5 MHz, ten frames match the
fixed-point model with the same statistics to 2.7e-5, 0.121 s per call
of three frames, 0.040 s/frame (0.047 s/frame with the host for 100
frames), 2.1 J/frame. The same kernel without the FC change failed
timing (hold) with three units.

## Smaller estimators by retraining (study)

`run_estimator_sweep` retrains SA-LSUN with fewer residual blocks R
and/or a narrower hidden layer (Width W, hidden = W x features), with
optional distillation from a trained network (`TeacherFile`), on three
GPU machines; `evaluate_heldout_sweep` evaluates every network on data
not used in training: later frames of the same simulation (551-700) and
two independent simulations with other sources (A: two sources; B: one
off-center source). The training-frame MSE is misleading (held-out MSE
4-15x the training MSE, the gap growing with the epochs), so the
comparison uses the held-out data. MSE relative to the current
estimators (R3 W2) trained for the same number of epochs (held-out
frames / A / B):

| epochs | estimators (with distillation) | multiply-adds | MSE ratio |
|---|---|---|---|
| 100 | R3 W1 (7 seeds) | 0.51 | 1.03 / 1.04 / 1.04 |
| 100 | R3 W0.5 | 0.27 | 1.16 / 1.13 / 1.17 |
| 200 | R6 W0.5 | 0.52 | 1.02 / 1.03 / 1.03 |
| 200 | R3 W1 | 0.51 | 1.09 / 1.08 / 1.09 |
| 200 | R3 W0.5 | 0.27 | 1.19 / 1.14 / 1.18 |
| 400 | R6 W0.5 | 0.52 | 1.04 / 0.99 / 1.01 |
| 400 | R3 W1.5 | 0.76 | 1.06 / 1.02 / 1.03 |
| 400 | R3 W1 | 0.51 | 1.10 / 1.04 / 1.07 |
| 400 | R3 W0.5 | 0.27 | 1.17 / 1.11 / 1.13 |
| 400 | R4 W0.5 (2 seeds) | 0.36 | 1.11 / 1.05 / 1.09 |
| 400 | R8 W0.25 (2 seeds) | 0.35 | 1.09 / 1.05 / 1.07 |
| 800 | R3 W1 (3 seeds) | 0.51 | 1.07 / 1.01 / 1.02 |
| 800 | R6 W0.5 | 0.52 | 1.00 / 1.00 / 1.00 |
| 800 | R4 W0.5 | 0.36 | 1.07 / 1.02 / 1.04 |

At 800 epochs the current estimators vary by seed as much as these
differences (held-out 6.19-7.01e-6, A 17.1-20.7e-6 over 3 seeds). R3 W1
trained for 1600 epochs (6.41 / 17.0 / 7.12e-6) is better than the
current estimators trained for 800 (6.56 / 18.3 / 7.67e-6).

Findings: halving the multiply-adds keeps the accuracy within a few
percent (R6 W0.5) to 10% (R3 W1); a quarter costs 11-19%. Distillation
(DistillWeight 20) removes the failed trainings of small networks (seed
spread about 1%); a weight of 100 does not help, nor does a better
teacher. One residual block is not enough, a 1 x 1 neighborhood (5x) and
a bottleneck projection of the features (2.7x on training frames) are
much worse. Longer training improves every configuration: the current
estimators reach held-out MSE 10.6e-6 (100 epochs, median), 8.4e-6
(200), 7.3e-6 (400) and 6.5e-6 (800), with no change of the hardware.
For the FPGA, R3 W1 keeps the LayerNorm work and halves the fully
connected layers and the GELU; R6 W0.5 is slightly more accurate but
doubles the LayerNorm and residual additions.

## Parametric replacements of the estimators (study)

Can a parametric angle estimator replace the learned one (and remove the
fully connected layers)? Frames 11-15, MSE relative to the trained
network:

| estimator | MSE / learned |
|---|---|
| linear map of the estimator features to the angles (least squares on frames 1-10) | 28 |
| local KLT (local block PCA) of the rotated channels, 3 x 3 blocks (`salsun2d_infer_klt`) | 49 |
| local KLT, 3 x 3 without the block itself | 95 |
| local KLT, 5 x 5 | 66 |
| local KLT of the block alone | 4.1 |
| constant angles (no adaptation) | 162 |

The local KLT orders the energy greedily stage by stage; it cannot
anticipate that only channels 1 and 9 survive the mask after the last
atom extension, which the learned estimators exploit end to end. Even
with the block itself as the only sample the greedy KLT is 4x worse.
So a training-free parametric estimator is not viable here; the
estimators can only be made cheaper by training smaller ones end to end.

## Fixed-point word lengths (simulated)

`evaluate_salsun2d_wordlength` runs the double-precision reference with
simulated rounding (`salsun2d_fixed_quantizer`, one power-of-two scaling
per tensor, or per channel for the block coefficients) on the trained
network (5 frames of 300 x 300). MSE relative to the double reference:

| bits | parameters only | signals only | both | both, coefficients per channel |
|---|---|---|---|---|
| 8  | 1.14 | 144 | 147 | – |
| 10 | 1.005 | 12.8 | 12.8 | 1.47 |
| 12 | 1.001 | 1.67 | 1.67 | 1.033 |
| 14 | 1.000 | 1.035 | 1.035 | 1.002 |
| 16 | 1.000 | 1.001 | 1.001 | 1.000 |

Sensitivity at 12 bits, one signal class at a time: block coefficients
1.66 (one scaling per tensor; the channels differ by orders of
magnitude), angles 1.006, everything else (rotation matrices,
standardized features, LayerNorm, hidden layer, parameters) <= 1.001.
So the estimators, which hold about 98% of the arithmetic, can use
12-bit signals and 10 to 12-bit weights, while the data path needs
14 to 16 bits with per-channel scaling.

## Numerical note

With the coefficient mask, rounding errors in the estimated angles do not
cancel between analysis and synthesis, and they grow by about 5x per stage.
With randomly perturbed parameters, the single-precision `dlnetwork` and the
single-precision reference both differ from the double-precision result by up
to about 7e-3 rad in the last estimator. Results in single precision (and
later in fixed point) are therefore judged by their error from the
double-precision reference (`salsun2d_cast_params(params,'double')`), not by
exact agreement with the GPU.

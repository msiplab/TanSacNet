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
| 3 | HDL Coder -> Vitis HLS, `sw_emu`, hardware build on temsip02 | whole-frame builds failed in routing (see Slack); streaming band design built (237.7 MHz) and validated on the U250: 10 frames match the reference to 8e-7; 90-lane engine failed timing (congestion); 30-lane engine with direct Givens rotations building |
| 4 | GPU training -> FPGA inference -> comparison in one script on temsip07 | |
| 5 | Fixed-point conversion and accuracy evaluation | word-length study done (below); HLS conversion pending |

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

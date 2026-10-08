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
| 3 | HDL Coder -> Vitis HLS, `sw_emu`, hardware build on temsip02 | whole-frame builds failed in routing (see Slack); streaming band design in sw_emu done, hardware build running |
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

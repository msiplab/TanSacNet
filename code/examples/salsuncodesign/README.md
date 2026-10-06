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
| 3 | HDL Coder -> Vitis HLS, `sw_emu`, hardware build on temsip02 | |
| 4 | GPU training -> FPGA inference -> comparison in one script on temsip07 | |
| 5 | Fixed-point conversion and accuracy evaluation | |

## Files

| File | Description |
|---|---|
| `salsun2d_extract_params.m` | Extracts all parameters from a `dlnetwork` (Mode `'Whole'`, ThetaMode `'Reuse'`, optional `Lv1_AcMask`) into a plain struct |
| `salsun2d_infer.m` | Reference inference: `[y,coefs,thetas] = salsun2d_infer(x,params)` |
| `salsun2d_cast_params.m` | Casts the parameters, e.g. to double for a high-precision reference |
| `Salsun2dInferTestCase.m` | Compares `salsun2d_infer` with `predict` of the `dlnetwork` |
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

## HLS version

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

## Numerical note

With the coefficient mask, rounding errors in the estimated angles do not
cancel between analysis and synthesis, and they grow by about 5x per stage.
With randomly perturbed parameters, the single-precision `dlnetwork` and the
single-precision reference both differ from the double-precision result by up
to about 7e-3 rad in the last estimator. Results in single precision (and
later in fixed point) are therefore judged by their error from the
double-precision reference (`salsun2d_cast_params(params,'double')`), not by
exact agreement with the GPU.

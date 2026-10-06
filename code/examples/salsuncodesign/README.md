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
| 2 | HLS-friendly rewrite (fixed sizes, loops, parameters as arguments) | |
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

```matlab
params = salsun2d_extract_params(reconnet);   % reconnet as in main_salsun2d.m
y = salsun2d_infer(single(x),params);         % same as reconnet.predict
runtests('Salsun2dInferTestCase')
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

## Numerical note

With the coefficient mask, rounding errors in the estimated angles do not
cancel between analysis and synthesis, and they grow by about 5x per stage.
With randomly perturbed parameters, the single-precision `dlnetwork` and the
single-precision reference both differ from the double-precision result by up
to about 7e-3 rad in the last estimator. Results in single precision (and
later in fixed point) are therefore judged by their error from the
double-precision reference (`salsun2d_cast_params(params,'double')`), not by
exact agreement with the GPU.

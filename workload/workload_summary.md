## PBE P90

| dataset | family | tokens | β | P90@1 | P90@2 | P90@4 | pred_speedup | actual_speedup |
|---|---|---|---|---|---|---|---|---|
| code_humaneval | code | 30,474 | 0.662 | 2 | 2 | 2 | 3.23 | 3.37 |
| code_mbpp | code | 21,344 | 0.696 | 2 | 2 | 2 | 3.34 | 3.42 |
| code_python_instruct | code | 719,801 | 0.708 | 2 | 2 | 2 | 3.38 | 2.96 |
| dialogue_alpaca | dialogue | 320,671 | 0.756 | 1 | 2 | 2 | 3.84 | 3.87 |
| dialogue_dolly | dialogue | 481,837 | 0.768 | 1 | 1 | 1 | 3.81 | 4.03 |
| dialogue_samsum | dialogue | 836,975 | 0.688 | 2 | 2 | 2 | 3.31 | 3.27 |
| math_gsm8k | math | 905,581 | 0.632 | 2 | 2 | 3 | 2.53 | 2.35 |
| math_math | math | 1,480,297 | 0.576 | 3 | 4 | 5 | 2.18 | 2.22 |
| math_metamath | math | 1,182,229 | 0.603 | 3 | 3 | 4 | 2.20 | 2.33 |
| nl_cnndailymail | nl | 4,239,428 | 0.671 | 2 | 2 | 2 | 3.26 | 3.11 |
| nl_openwebtext | nl | 5,403,154 | 0.688 | 2 | 2 | 2 | 3.31 | 3.53 |
| nl_wikitext103 | nl | 516,825 | 0.746 | 1 | 2 | 2 | 3.81 | 3.74 |

## PBE mean

| dataset | family | mean@1 | mean@2 | mean@4 |
|---|---|---|---|---|
| code_humaneval | code | 1.27 | 1.36 | 1.47 |
| code_mbpp | code | 1.27 | 1.35 | 1.45 |
| code_python_instruct | code | 1.36 | 1.47 | 1.62 |
| dialogue_alpaca | dialogue | 1.20 | 1.25 | 1.29 |
| dialogue_dolly | dialogue | 1.16 | 1.19 | 1.22 |
| dialogue_samsum | dialogue | 1.41 | 1.53 | 1.60 |
| math_gsm8k | math | 1.56 | 1.75 | 1.91 |
| math_math | math | 1.88 | 2.27 | 2.71 |
| math_metamath | math | 1.68 | 1.98 | 2.33 |
| nl_cnndailymail | nl | 1.35 | 1.45 | 1.52 |
| nl_openwebtext | nl | 1.30 | 1.37 | 1.41 |
| nl_wikitext103 | nl | 1.21 | 1.26 | 1.31 |

## Speedup model

**Purpose.** Curve fitting here learns an empirical mapping from workload
characteristics (Heaps' β and PBE P90 at n=1,2,4) to the speedup of n-gram
speculative decoding, so we can *predict* a workload's speedup before running the
full benchmark: `pred_speedup = f(β, P90@1, P90@2, P90@4)`.

> จุดประสงค์: เรียนความสัมพันธ์เชิงประจักษ์ระหว่างลักษณะ workload (β, PBE P90 ที่ n=1,2,4)
> กับ speedup จริงของ n-gram speculative decoding เพื่อ *ทำนาย* speedup ของ workload ใหม่
> ก่อนรัน benchmark จริง

**Pipeline.**
```
dataset → measure β, P90@1, P90@2, P90@4
        → run n-gram spec decoding → actual_speedup
        → fit f() (least squares) → pred_speedup → residual (MAE/RMSE/R²)
```

**pred_speedup (linear regression).** A least-squares fit of measured
`actual_speedup` on the workload features: `pred_speedup = w0 + w1·β + w2·P90@1 +
w3·P90@2 + w4·P90@4`. Features enter flat (no product chain), so no unmeasured
depth is required. β carries most of the signal.

**Fit to actual_speedup — linear:**
```
fit: actual_speedup ~ features (12 datasets, linear least squares, in-sample)

  [β, P90@1, P90@2, P90@4] R²=0.908 MAE=0.143 RMSE=0.178
      actual_speedup = +3.200·β + -0.311·P90@1 + +0.673·P90@2 + -0.603·P90@4 +1.593
      per-family MAE: code=0.21  dialogue=0.10  math=0.12  nl=0.15
  [β, P90@1, P90@2       ] R²=0.853 MAE=0.160 RMSE=0.224
      actual_speedup = +9.465·β + -0.007·P90@1 + -0.000·P90@2 -3.264
      per-family MAE: code=0.32  dialogue=0.03  math=0.16  nl=0.12
  [β, P90@1              ] R²=0.853 MAE=0.160 RMSE=0.224
      actual_speedup = +9.468·β + -0.008·P90@1 -3.266
      per-family MAE: code=0.32  dialogue=0.03  math=0.16  nl=0.12
  [β                     ] R²=0.853 MAE=0.160 RMSE=0.224
      actual_speedup = +9.548·β -3.335
      per-family MAE: code=0.32  dialogue=0.03  math=0.16  nl=0.12
```

**Fit to actual_speedup — polynomial (degree 2):**
```
poly fit: actual_speedup (12 datasets, in-sample)

  [β, β²              ]  R²=0.854 MAE=0.158 RMSE=0.223
      actual_speedup = +17.946·β -6.221·β² -6.149
      per-family MAE: code=0.31  dialogue=0.03  math=0.18  nl=0.11
  [β, P90@1, +cross  ]  R²=0.965 MAE=0.083 RMSE=0.109
      actual_speedup = +746.424·β +66.287·P90@1 -434.869·β² -3.641·P90@1² -77.181·β·P90@1 -316.141
      per-family MAE: code=0.12  dialogue=0.06  math=0.02  nl=0.14
```
β carries most of the signal; polynomial adds little over linear given only 12 datasets.

**Evaluation & improvement.** `residual = pred_speedup − actual_speedup`. A high
residual is **feedback to refine the estimator, not a failed experiment**: break it
down per family (code/math/dialogue/nl) to see whether one function suffices or a
`family` term is needed; check whether the 4 features are enough (else add
acceptance_rate / repetition_rate as a later stage); compare model variants and keep
the lowest MAE/RMSE. Re-run with real (non-simulation) `actual_speedup` to validate.

**pred vs actual_speedup (simulation):**

| dataset | pred_speedup | actual_speedup | residual |
|---|---|---|---|
| code_humaneval | 3.23 | 3.37 | -0.14 |
| code_mbpp | 3.34 | 3.42 | -0.08 |
| code_python_instruct | 3.38 | 2.96 | +0.42 |
| dialogue_alpaca | 3.84 | 3.87 | -0.03 |
| dialogue_dolly | 3.81 | 4.03 | -0.22 |
| dialogue_samsum | 3.31 | 3.27 | +0.04 |
| math_gsm8k | 2.53 | 2.35 | +0.18 |
| math_math | 2.18 | 2.22 | -0.04 |
| math_metamath | 2.20 | 2.33 | -0.13 |
| nl_cnndailymail | 3.26 | 3.11 | +0.15 |
| nl_openwebtext | 3.31 | 3.53 | -0.21 |
| nl_wikitext103 | 3.81 | 3.74 | +0.07 |

MAE=0.143  RMSE=0.178
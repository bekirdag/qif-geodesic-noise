# 2026-02-01 GPU Validation Suite (RTX 3090)

Source: user-reported runs on Ubuntu (RTX 3090, CUDA 12.6, CuPy 13.6.0).

## Sensitivity sweep (2048 s, r=1, fit+phi)

- max_bins=128: LR = -7.529139e+04 (all groups)
- max_bins=256: LR = -1.893584e+05 (all groups)
- max_bins=512: LR = -3.568421e+05 (all groups)

Logs: `test-runs/20260201-171426_sensitivity_sweep/`

## Bootstrap (128 s, 128 bins, BBH_snr_306)

- LR = -4.640351e+03
- p = 1.0000 (n=50)

Log: `test-runs/20260201-184235_bootstrap/bootstrap.log`

## Bootstrap (128 s, 128 bins, all groups, n=20)

- LR = -4.640351e+03 across all groups
- p = 1.0000 for all groups (n=20)

Log: `test-runs/20260201-191634_bootstrap/bootstrap.log`

## Stress tests (128 s, 128 bins, BBH_snr_306)

- baseline LR = -4.640351e+03
- rank-2 LR = -4.640351e+03
- phi_fixed LR = -4.640351e+03

Log: `test-runs/20260201-184649_stress_tests/stress_tests.log`

## Line mask + transfer (template) (128 s, 128 bins, BBH_snr_306)

- LR = -4.599273e+03

Log: `test-runs/20260201-185807_mask_transfer/mask_transfer.log`

## Synthetic injection recovery

Command:
```
./scripts/run_injection_test.py --alpha 1.0 --n-f 16 --m-eff 20
```

Result:
- lnL_env = 5.727279e+03
- lnL_qif = 5.830983e+03
- LR = 2.074083e+02 (positive)

Interpretation: the pipeline can recover an injected QIF signal under controlled synthetic conditions.

# 2026-02-01 GPU Full-Duration Run (RTX 3090)

Source: user-reported run on Ubuntu (RTX 3090, CUDA 12.6, CuPy 13.6.0).
Command:

```
python qif_v2_cuda.py \
  --data-root data \
  --device auto \
  --max-seconds 2048 \
  --max-bins 512 \
  --fit --fit-phi --max-iter 120 --n-starts 2
```

Output:

```
Backend: cupy
+-------------+------------+------+---+---------+--------------+--------------+---------------+---+------+
| DATASET     | GPS        | BINS | R | FIT_PHI | LNL_ENV      | LNL_QIF      | LR            | P | P_SE |
+=============+============+======+===+=========+==============+==============+===============+===+======+
| BBH_snr_306 | 1000245760 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BBH_snr_344 | 1002150400 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BBH_snr_379 | 1001558528 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BBH_snr_387 | 1001622016 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BBH_snr_587 | 1001619968 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BNS_snr_379 | 1001329152 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BNS_snr_379 | 1001331200 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
| BNS_snr_379 | 1001333248 | 512  | 1 | yes     | 4.466109e+07 | 4.448267e+07 | -3.568421e+05 |   |      |
+-------------+------------+------+---+---------+--------------+--------------+---------------+---+------+
```

Notes:
- LR is negative for all groups under this configuration.
- Identical LR values across groups suggests limited sensitivity under current settings.


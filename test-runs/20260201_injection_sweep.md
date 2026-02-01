# 2026-02-01 Synthetic Injection Sweep

Command(s):

```
./scripts/run_injection_test.py --alpha 0.1 --n-f 16 --m-eff 20
./scripts/run_injection_test.py --alpha 0.3 --n-f 16 --m-eff 20
./scripts/run_injection_test.py --alpha 1.0 --n-f 16 --m-eff 20
```

Results:

- alpha=0.1: lnL_env=7.937760e+03, lnL_qif=8.041397e+03, LR=2.072729e+02
- alpha=0.3: lnL_env=6.883093e+03, lnL_qif=6.986764e+03, LR=2.073435e+02
- alpha=1.0: lnL_env=5.727279e+03, lnL_qif=5.830983e+03, LR=2.074083e+02

Notes:
- Positive LR across all injected amplitudes demonstrates recoverability under synthetic conditions.
- LR is nearly constant here; this sweep confirms positive detection but does not establish a threshold.


| Model | Rate | Burst | Policy | Reps | Success rps (min–max) | p50 / p95 / p99 ms | Unsuccessful % | Client-invalid runs | Target met every run |
| --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| custom | 200 | False | immediate | 3 | 200.0 (200.0–200.0) | 2.02 / 3.04 / 3.58 | 0.00 | 0 | True |
| custom | 200 | False | single | 3 | 199.9 (199.6–200.0) | 2.02 / 2.92 / 3.57 | 0.07 | 1 | False |
| custom | 200 | False | timed | 3 | 200.0 (200.0–200.0) | 7.90 / 9.31 / 9.74 | 0.00 | 0 | True |
| custom | 800 | False | immediate | 3 | 800.0 (800.0–800.0) | 0.98 / 1.12 / 1.50 | 0.00 | 0 | True |
| custom | 800 | False | single | 3 | 800.0 (800.0–800.0) | 0.99 / 1.15 / 1.56 | 0.00 | 0 | True |
| custom | 800 | False | timed | 3 | 498.1 (467.6–531.1) | 269.51 / 1448.91 / 2639.42 | 37.21 | 3 | False |
| custom | 1600 | False | immediate | 3 | 561.1 (544.8–582.2) | 297.36 / 1310.92 / 2102.30 | 64.67 | 3 | False |
| custom | 1600 | False | single | 3 | 553.6 (521.4–579.9) | 311.07 / 1305.60 / 2071.29 | 65.13 | 3 | False |
| custom | 1600 | False | timed | 3 | 447.6 (388.4–494.3) | 374.92 / 1693.25 / 2956.98 | 71.76 | 3 | False |
| huggingface | 20 | False | immediate | 3 | 20.0 (20.0–20.0) | 11.00 / 30.84 / 31.39 | 0.06 | 1 | False |
| huggingface | 20 | False | single | 3 | 20.0 (20.0–20.0) | 10.45 / 30.79 / 31.37 | 0.00 | 0 | True |
| huggingface | 20 | False | timed | 3 | 20.0 (20.0–20.0) | 17.58 / 37.05 / 37.55 | 0.00 | 0 | True |
| huggingface | 80 | False | immediate | 3 | 66.2 (52.7–80.0) | 260.35 / 538.79 / 552.28 | 16.63 | 0 | False |
| huggingface | 80 | False | single | 3 | 80.0 (80.0–80.0) | 16.55 / 27.20 / 30.80 | 0.00 | 0 | True |
| huggingface | 80 | False | timed | 3 | 61.1 (57.3–66.2) | 489.61 / 795.89 / 811.26 | 22.80 | 0 | False |
| huggingface | 160 | False | immediate | 3 | 49.4 (49.3–49.6) | 785.20 / 816.71 / 841.19 | 68.71 | 0 | False |
| huggingface | 160 | False | single | 3 | 93.0 (92.6–93.4) | 352.57 / 398.50 / 419.61 | 41.57 | 0 | False |
| huggingface | 160 | False | timed | 3 | 49.2 (48.8–49.4) | 786.35 / 822.33 / 913.10 | 68.83 | 1 | False |

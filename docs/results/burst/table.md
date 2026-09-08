| Model | Rate | Burst | Policy | Reps | Success rps (min–max) | p50 / p95 / p99 ms | Unsuccessful % | Client-invalid runs | Target met every run |
| --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| custom | 800 | False | immediate | 3 | 625.6 (535.0–800.0) | 140.53 / 876.84 / 1455.90 | 20.39 | 2 | False |
| custom | 800 | False | single | 3 | 718.4 (555.3–800.0) | 108.22 / 398.39 / 620.67 | 9.48 | 1 | False |
| custom | 800 | False | timed | 3 | 602.4 (556.6–686.1) | 162.27 / 1051.18 / 1757.82 | 22.57 | 3 | False |
| custom | 800 | True | immediate | 3 | 492.9 (429.7–527.6) | 239.61 / 645.78 / 1118.84 | 38.39 | 3 | False |
| custom | 800 | True | single | 3 | 448.7 (325.9–532.8) | 251.07 / 1037.60 / 2054.69 | 43.56 | 3 | False |
| custom | 800 | True | timed | 3 | 459.1 (397.2–519.7) | 277.89 / 806.94 / 1183.17 | 42.46 | 3 | False |
| huggingface | 80 | False | immediate | 3 | 79.9 (79.9–79.9) | 16.42 / 28.33 / 29.73 | 0.00 | 0 | True |
| huggingface | 80 | False | single | 3 | 79.9 (79.9–79.9) | 17.16 / 28.56 / 32.24 | 0.00 | 0 | True |
| huggingface | 80 | False | timed | 3 | 62.1 (49.9–79.9) | 516.18 / 567.95 / 586.57 | 20.36 | 0 | False |
| huggingface | 80 | True | immediate | 3 | 47.4 (47.3–47.5) | 628.80 / 812.30 / 847.16 | 38.83 | 0 | False |
| huggingface | 80 | True | single | 3 | 79.9 (79.7–80.0) | 180.03 / 332.25 / 360.55 | 0.14 | 0 | False |
| huggingface | 80 | True | timed | 3 | 47.3 (47.3–47.3) | 628.52 / 826.79 / 848.11 | 39.06 | 0 | False |

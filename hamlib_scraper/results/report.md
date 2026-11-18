# HamLib chemistry term-count scaling

33 families and 3,678 encoded Hamiltonians.

3 families met the 6-size minimum for screening. 0 passed the screen.

A candidate must reduce both holdout errors by at least 10%. Each error is compared with the best polynomial-scaling baseline for that metric.

Fits use the median positive term count at each qubit size. Each size has equal weight.

[HamLib chemistry metadata](https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/)

| Family | Hamiltonians | Qubit sizes | Fitted function | Log R² | CV log RMSE | Largest-size log error | Status |
|---|---:|---:|---|---:|---:|---:|---|
| electronic/bond_breaking | 297 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H10_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H10_pyramid | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H10_ring | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H10_sheet | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H12_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H12_pyramid | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H12_ring | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H12_sheet | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H14_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H14_pyramid | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H14_ring | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H14_sheet | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H16_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H16_pyramid | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H16_ring | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H16_sheet | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H18_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H20_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H24_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H28_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H2_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H32_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H36_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H40_linear | 3 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H4_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H50_linear | 3 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H60_linear | 3 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H6_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/hydrogen_data/H8_linear | 48 | 1 | — | — | — | — | One positive-term qubit size |
| electronic/standard | 270 | 18 | `T(n) = 3911.29 x^(4.47759) exp(0.513771 (1/x - 1)), x = n / 20.6619` | 0.995 | 0.261 | 0.235 | No super-polynomial signal |
| electronic/transition_metals | 165 | 18 | `T(n) = 256779 x^(3.287), x = n / 50.3009` | 0.966 | 0.316 | 0.0248 | No super-polynomial signal |
| vibrational | 1,689 | 70 | `T(n) = 334.423 x^(1.48895), x = n / 38.8878` | 0.196 | 3.47 | 9.01 | No super-polynomial signal |

Cross-validation holds out one size at a time. Largest-size error is the absolute log prediction error when training on smaller sizes only.

The vibrational directory, when present, pools molecules, encodings, and basis truncations.

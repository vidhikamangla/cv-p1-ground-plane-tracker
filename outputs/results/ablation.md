| row | change | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | fps |
|---|---|---|---|---|---|---|---|---|---|
| T0 | image / IoU / 1-stage (SORT-like) | 26.2 | 43.88 | 15.72 | 22.45 | 21.54 | 655 | 858 | - |
| T1 | + CMC | 26.86 | 43.69 | 16.59 | 22.94 | 23.14 | 716 | 1035 | - |
| T2 | + two-stage association | 28.26 | 44.54 | 18.05 | 16.97 | 23.73 | 876 | 940 | - |
| T3 | Mahalanobis cost (still image space) | 27.81 | 37.3 | 20.81 | -21.83 | 25.53 | 1382 | 788 | - |
| T4 | ground plane, H_t = H_ref @ W(t->ref) | 23.86 | 36.68 | 15.65 | -28.31 | 19.11 | 2394 | 925 | - |
| T5 | + jersey-colour gate | 26.76 | 39.95 | 18.05 | -12.46 | 22.36 | 2474 | 907 | - |
| T7 | ground plane + CMC, cost held at IoU (isolates the space) | 27.78 | 43.33 | 17.95 | 13.26 | 23.6 | 1075 | 1129 | - |
| T8 | ground + CMC + per-detection R = J Rpx J^T | 21.48 | 38.44 | 12.14 | -20.32 | 16.0 | 2734 | 1107 | - |
| T6 | ground plane, STATIC H_ref, no CMC | 31.24 | 39.89 | 24.67 | -11.48 | 27.22 | 1620 | 656 | - |
| T9 | ground + ADAPTIVE CMC (apply the warp only when the camera moved) | 28.07 | 39.72 | 19.99 | -11.65 | 23.79 | 2059 | 879 | - |

All rows share one frozen detection cache (yolo26n, COCO weights, conf 0.10) so differences are tracking, not detection. Both coordinate spaces were tuned the same way: an accel_std x meas_std sweep on clip A, best taken (image 250 px/s^2 / 20 px; ground 40 m/s^2 / 0.35 m). Read the per-sequence CSVs - the combined column hides the CMC effect almost perfectly.

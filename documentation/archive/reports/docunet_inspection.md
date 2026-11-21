# DocUNet dataset inspection

Inspected: 2026-11-10T05:54:48.858032+00:00

## Archive status

| component   | source                     | archive_size   | status   | details                            |
|:------------|:---------------------------|:---------------|:---------|:-----------------------------------|
| original    | original project server    | 328.5 MiB      | ready    | 130 extracted; 0 reused; 0 skipped |
| crop        | author Google Drive backup | 281.1 MiB      | ready    | 130 extracted; 0 reused; 0 skipped |
| scan        | original project server    | 416.2 MiB      | ready    | 65 extracted; 0 reused; 0 skipped  |
| eval        | author Google Drive backup | 1.3 MiB        | ready    | 45 extracted; 0 reused; 0 skipped  |

## Image inventory

| group    |   images | total_size   | image_formats   | common_resolutions                           |   unreadable |
|:---------|---------:|:-------------|:----------------|:---------------------------------------------|-------------:|
| original |      130 | 332.7 MiB    | .jpg            | 3024×4032 (100), 4032×3024 (30)              |            0 |
| crop     |      130 | 286.0 MiB    | .png            | 1724×1241 (1), 1632×817 (1), 1621×1675 (1)   |            0 |
| scan     |       65 | 419.5 MiB    | .png            | 2550×3300 (17), 1265×1770 (1), 2506×1262 (1) |            0 |

## Correspondence checks

- Crop → original: 0/130
- Photo → scan: 260/260
- Scan documents: 65

## Evaluation code files

- `eval/SIFTflow/SIFTflowc2f.m`
- `eval/SIFTflow/computeColor.m`
- `eval/SIFTflow/demo.m`
- `eval/SIFTflow/flowToColor.m`
- `eval/SIFTflow/mexDenseSIFT/Matrix.cpp`
- `eval/SIFTflow/mexDenseSIFT/Vector.cpp`
- `eval/SIFTflow/mexDenseSIFT/dir.cpp`
- `eval/SIFTflow/mexDenseSIFT/mexDenseSIFT.cpp`
- `eval/SIFTflow/mexDenseSIFT/mexDenseSIFT.m`
- `eval/SIFTflow/mexDenseSIFT/showColorSIFT.m`
- `eval/SIFTflow/mexDiscreteFlow/BPFlow.cpp`
- `eval/SIFTflow/mexDiscreteFlow/Stochastic.cpp`
- `eval/SIFTflow/mexDiscreteFlow/main.cpp`
- `eval/SIFTflow/mexDiscreteFlow/mexDiscreteFlow.cpp`
- `eval/SIFTflow/mexDiscreteFlow/mexDiscreteFlow.m`
- `eval/SIFTflow/warpFL.m`
- `eval/SIFTflow/warpFLColor.m`
- `eval/SIFTflow/warpImage.m`
- `eval/evalUnwarp.m`

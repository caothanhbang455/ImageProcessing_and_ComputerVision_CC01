# Document Unwarping – Dataset EDA Report

Updated (UTC): 2026-11-20T01:51:11.234460+00:00

This document contains measured dataset statistics, figure companion tables, and explicit limitations. It is generated incrementally by `[20.11] Dataset_EDA.ipynb` and does not claim model-performance results.

- Image statistics: 512px longest side for pixel-based descriptors.
- UVDoc Final pixel sample limit: 2,400 (metadata inventory covers all records).
- Random seed: 20261120.
- All method-feasibility descriptors are **proxies**, not validated detectors.

## Completed analyses

### 01.06 Inventory readiness
All listed datasets must pass Notebook 1 structural checks. Small path probes verify the current runtime; full decoding occurs in later stages.

**Readiness** · full data: `dataset/_metadata/01_06_readiness.csv`

| dataset | rows | expected | audit_ready | sample_paths_checked | missing_sample_paths |
| --- | --- | --- | --- | --- | --- |
| UVDoc Final | 20000 | 20000 | True | 12 | 0 |
| UVDoc Benchmark | 50 | 50 | True | 12 | 0 |
| DocUNet | 130 | 130 | True | 12 | 0 |
| DIR300 | 300 | 300 | True | 12 | 0 |

### 02.02 Source composition
Base-geometry grouping removes the final two underscore-separated suffixes. It is supported here by the observed four-variant pattern; do not equate variants with physically independent captures.

**Dataset composition** · full data: `dataset/_metadata/02_02_dataset_composition.csv`

| dataset | images | geometry_variants | base_geometries | textures | backgrounds | invalid_annotations |
| --- | --- | --- | --- | --- | --- | --- |
| UVDoc Final | 20000 | 4032 | 1008 | 3003 | 5470 | 0 |
| UVDoc Benchmark | 50 | 50 | 50 | 50 | 50 | 0 |

**Geometry variant multiplicity** · full data: `dataset/_metadata/02_02_geometry_variant_multiplicity.csv`

| variants_per_base | base_geometry_count |
| --- | --- |
| 4 | 1008 |

**Renderings per base geometry** · full data: `dataset/_metadata/02_02_renderings_per_base_geometry.csv`

| statistic | renders |
| --- | --- |
| count | 1008.0 |
| mean | 19.84127 |
| std | 0.78124 |
| min | 16.0 |
| 25% | 20.0 |
| 50% | 20.0 |
| 75% | 20.0 |
| max | 20.0 |

### 02.03 Renderings per base geometry
Each bar counts independently named base geometries with that many rendered images.

Figure: `runs/dataset_eda/figures/02_03.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/02_03_figure_data.csv`

| renderings_per_base | base_count |
| --- | --- |
| 16 | 40 |
| 20 | 968 |

### 02.04 Cross-release source overlap
Identical `sample_id` values are local to each release and are **not** evidence of a duplicate image. Shared texture filenames indicate a possible repeated content source, not proof of identical pixels. A geometry-disjoint benchmark does not guarantee texture-disjoint content.

**Overlap counts** · full data: `dataset/_metadata/02_04_overlap_counts.csv`

| identity | final_unique | benchmark_unique | shared_identifiers | examples |
| --- | --- | --- | --- | --- |
| sample_id | 20000 | 50 | 50 | 00000, 00001, 00002, 00003, 00004, 00005, 00006, 00007, 00008, 00009, 00010, 00011 |
| geom_name | 4032 | 50 | 0 |  |
| base_geometry | 1008 | 50 | 0 |  |
| texture_name | 3003 | 50 | 12 | 04-01.png, 04-02.png, 04-05.png, 04-06.png, 04-11.png, 04-15.png, 05-01.png, 05-03.png, 05-04.png, 05-06.png, 05-08.png, 05-09.png |
| background_name | 5470 | 50 | 0 |  |

### 03.03 Image representation
All images are checked at header level. EXIF orientation is recorded, not used to overwrite originals. Aspect ratio and pixel count are not measures of geometric distortion.

**Resolution summary** · full data: `dataset/_metadata/03_03_resolution_summary.csv`

| dataset | image_role | count | width_min | width_median | width_p95 | height_median | mp_median | mp_p95 | aspect_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIR300 | dist | 300 | 2041 | 3024.0 | 3472.0 | 4032.0 | 12.19277 | 16.05453 | 0.75 |
| DIR300 | gt | 300 | 1336 | 1634.0 | 2028.0 | 2352.0 | 3.83518 | 5.79318 | 0.70295 |
| DocUNet | crop | 130 | 654 | 1339.0 | 1789.85 | 1718.0 | 2.26532 | 3.80057 | 0.78302 |
| DocUNet | original | 130 | 3024 | 3024.0 | 4032.0 | 4032.0 | 12.19277 | 12.19277 | 0.75 |
| DocUNet | scan | 65 | 677 | 2398.0 | 2550.0 | 3190.0 | 7.47857 | 8.415 | 0.7508 |
| UVDoc Benchmark | rendered | 50 | 785 | 908.0 | 1025.95 | 1217.0 | 1.13135 | 1.31314 | 0.74302 |
| UVDoc Final | rendered | 20000 | 488 | 488.0 | 488.0 | 712.0 | 0.34746 | 0.34746 | 0.68539 |

**Modes and EXIF** · full data: `dataset/_metadata/03_03_modes_and_exif.csv`

| dataset | image_role | mode | exif_orientation | count |
| --- | --- | --- | --- | --- |
| DIR300 | dist | RGB | 1 | 298 |
| DIR300 | dist | RGBA | 1 | 2 |
| DIR300 | gt | RGB | 1 | 299 |
| DIR300 | gt | RGBA | 1 | 1 |
| DocUNet | crop | RGB | 1 | 130 |
| DocUNet | original | RGB | 1 | 100 |
| DocUNet | original | RGB | 6 | 30 |
| DocUNet | scan | RGB | 1 | 38 |
| DocUNet | scan | RGBA | 1 | 27 |
| UVDoc Benchmark | rendered | RGB | 1 | 50 |
| UVDoc Final | rendered | RGB | 1 | 20000 |

**Failed image headers** · full data: `dataset/_metadata/03_03_failed_image_headers.csv`

_No rows available._

### 03.04 Image size distribution
Absolute counts reflect different dataset sizes; compare proportions with the full CSV before drawing conclusions.

Figure: `runs/dataset_eda/figures/03_04.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/03_04_figure_data.csv`

| dataset | mp_bin | count |
| --- | --- | --- |
| DIR300 | 0–0.2 | 0 |
| DIR300 | 0.2–0.5 | 0 |
| DIR300 | 0.5–1 | 0 |
| DIR300 | 1–2 | 0 |
| DIR300 | 2–4 | 0 |
| DIR300 | 4–8 | 29 |
| DIR300 | 8–16 | 254 |
| DIR300 | 16–inf | 17 |
| DocUNet | 0–0.2 | 0 |
| DocUNet | 0.2–0.5 | 0 |
| DocUNet | 0.5–1 | 0 |
| DocUNet | 1–2 | 42 |
| DocUNet | 2–4 | 84 |
| DocUNet | 4–8 | 4 |
| DocUNet | 8–16 | 0 |
| DocUNet | 16–inf | 0 |
| UVDoc Benchmark | 0–0.2 | 0 |
| UVDoc Benchmark | 0.2–0.5 | 0 |
| UVDoc Benchmark | 0.5–1 | 5 |
| UVDoc Benchmark | 1–2 | 45 |
| UVDoc Benchmark | 2–4 | 0 |
| UVDoc Benchmark | 4–8 | 0 |
| UVDoc Benchmark | 8–16 | 0 |
| UVDoc Benchmark | 16–inf | 0 |
| UVDoc Final | 0–0.2 | 0 |

_Showing 25 of 32 rows; full CSV is saved separately._

### 03.05 Photometric descriptors
Pixel-level quantities are computed on images downscaled to at most 512px per side. UVDoc Final uses a seeded 2,400-image sample; other benchmark inputs are fully processed. Laplacian variance is a scale-dependent sharpness proxy, tile luminance range an illumination proxy, and edge fraction not a document-boundary detector.

**Photometric dataset summary** · full data: `dataset/_metadata/03_05_photometric_dataset_summary.csv`

| dataset | luminance_mean_count | luminance_mean_median | luminance_mean_mean | luminance_mean_std | luminance_std_count | luminance_std_median | luminance_std_mean | luminance_std_std | saturation_mean_count | saturation_mean_median | saturation_mean_mean | saturation_mean_std | laplacian_variance_count | laplacian_variance_median | laplacian_variance_mean | laplacian_variance_std | edge_fraction_count | edge_fraction_median | edge_fraction_mean | edge_fraction_std | tile_luminance_range_count | tile_luminance_range_median | tile_luminance_range_mean | tile_luminance_range_std | dark_fraction_count | dark_fraction_median | dark_fraction_mean | dark_fraction_std | bright_fraction_count | bright_fraction_median | bright_fraction_mean | bright_fraction_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIR300 | 300 | 132.7349 | 132.4644 | 18.2613 | 300 | 56.2185 | 56.1921 | 13.3875 | 300 | 55.2383 | 56.0096 | 26.6537 | 300 | 580.0044 | 648.1481 | 370.431 | 300 | 0.0991 | 0.1043 | 0.0436 | 300 | 96.2169 | 97.9686 | 29.1632 | 300 | 0.0654 | 0.1191 | 0.1325 | 300 | 0.0 | 0.0063 | 0.0165 |
| DocUNet | 130 | 164.9636 | 161.7357 | 22.3675 | 130 | 54.2676 | 53.7862 | 14.3344 | 130 | 50.4211 | 51.8515 | 17.9118 | 130 | 998.5561 | 1071.867 | 469.1868 | 130 | 0.1291 | 0.1317 | 0.0447 | 130 | 79.2834 | 82.4913 | 31.1332 | 130 | 0.0557 | 0.0858 | 0.0979 | 130 | 0.0016 | 0.0507 | 0.1005 |
| UVDoc Benchmark | 50 | 146.8137 | 144.7798 | 17.8005 | 50 | 39.5943 | 39.1037 | 8.726 | 50 | 66.2044 | 73.4287 | 29.3482 | 50 | 522.9178 | 652.0986 | 526.7526 | 50 | 0.1112 | 0.1031 | 0.0375 | 50 | 70.8179 | 74.143 | 24.9859 | 50 | 0.0102 | 0.0219 | 0.0275 | 50 | 0.0 | 0.0083 | 0.0195 |
| UVDoc Final | 2400 | 125.6757 | 121.3586 | 30.8847 | 2400 | 43.5728 | 44.3284 | 11.9738 | 2400 | 56.2751 | 65.4621 | 48.1753 | 2400 | 486.3063 | 645.5647 | 530.2311 | 2400 | 0.129 | 0.1335 | 0.0574 | 2400 | 65.4945 | 68.7071 | 27.0203 | 2400 | 0.0657 | 0.1332 | 0.1961 | 2400 | 0.0 | 0.0138 | 0.0476 |

**Pixel decoding errors** · full data: `dataset/_metadata/03_05_pixel_decoding_errors.csv`

_No rows available._

### 03.06 Cross-dataset appearance summary
The plotted median luminance and additional companion columns describe inputs, not method performance.

Figure: `runs/dataset_eda/figures/03_06.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/03_06_figure_data.csv`

| dataset | n | luminance_median | contrast_median | edge_fraction_median | tile_range_median |
| --- | --- | --- | --- | --- | --- |
| DIR300 | 300 | 132.73492 | 56.21854 | 0.09908 | 96.21686 |
| DocUNet | 130 | 164.96359 | 54.26761 | 0.12913 | 79.2834 |
| UVDoc Benchmark | 50 | 146.81371 | 39.59435 | 0.11122 | 70.81786 |
| UVDoc Final | 2400 | 125.67575 | 43.57277 | 0.12901 | 65.49452 |

### 04.02 Representative input gallery
Figure: `runs/dataset_eda/galleries/representative_inputs.jpg`. The companion table records every pictured sample ID and source path. Visual content requires manual review; do not infer page curvature from image IDs.

**Gallery identity and paths** · full data: `dataset/_metadata/04_02_gallery_identity_and_paths.csv`

| dataset | sample_id | input_path | reference_path |
| --- | --- | --- | --- |
| UVDoc Final | 05237 | dataset/UVDoc/UVDoc_final/img/05237.png |  |
| UVDoc Final | 06921 | dataset/UVDoc/UVDoc_final/img/06921.png |  |
| UVDoc Final | 13717 | dataset/UVDoc/UVDoc_final/img/13717.png |  |
| UVDoc Final | 17556 | dataset/UVDoc/UVDoc_final/img/17556.png |  |
| UVDoc Benchmark | 00011 | dataset/UVDoc/UVDoc_benchmark/img/00011.png |  |
| UVDoc Benchmark | 00020 | dataset/UVDoc/UVDoc_benchmark/img/00020.png |  |
| UVDoc Benchmark | 00025 | dataset/UVDoc/UVDoc_benchmark/img/00025.png |  |
| UVDoc Benchmark | 00036 | dataset/UVDoc/UVDoc_benchmark/img/00036.png |  |
| DocUNet | 13_2 | dataset/DocUNet/crop/13_2 copy.png | dataset/DocUNet/scan/13.png |
| DocUNet | 38_1 | dataset/DocUNet/crop/38_1 copy.png | dataset/DocUNet/scan/38.png |
| DocUNet | 46_2 | dataset/DocUNet/crop/46_2 copy.png | dataset/DocUNet/scan/46.png |
| DocUNet | 59_2 | dataset/DocUNet/crop/59_2 copy.png | dataset/DocUNet/scan/59.png |
| DIR300 | 38 | dataset/DIR300/_source/DIR300/dist/38.png | dataset/DIR300/_source/DIR300/gt/38.png |
| DIR300 | 118 | dataset/DIR300/_source/DIR300/dist/118.png | dataset/DIR300/_source/DIR300/gt/118.png |
| DIR300 | 239 | dataset/DIR300/_source/DIR300/dist/239.png | dataset/DIR300/_source/DIR300/gt/239.png |
| DIR300 | 272 | dataset/DIR300/_source/DIR300/dist/272.png | dataset/DIR300/_source/DIR300/gt/272.png |

### 04.03 Classical cue probes
OpenCV Canny/Hough and largest-contour descriptors are unvalidated probes; Hough lines may be paper edges or graphical elements rather than text lines. A four-vertex contour is **not** verified page-corner detection. No classical unwarping is implemented.

**Probe summary** · full data: `dataset/_metadata/04_03_probe_summary.csv`

| dataset | n | median_edges | median_segments | median_long_segments | median_contour_area |
| --- | --- | --- | --- | --- | --- |
| DIR300 | 300 | 0.1016 | 253.0 | 100.0 | 0.51591 |
| DocUNet | 130 | 0.13329 | 357.0 | 121.5 | 0.06819 |
| UVDoc Final | 240 | 0.12759 | 298.0 | 86.0 | 0.00886 |

**Probe errors** · full data: `dataset/_metadata/04_03_probe_errors.csv`

_No rows available._

### 04.04 Classical cue gallery and review template
Gallery: `runs/dataset_eda/galleries/classical_probe_gallery.jpg`; manual worksheet: `dataset/_metadata/classical_manual_review_template.csv`. The worksheet leaves all qualitative judgments blank for human review. The companion table identifies each displayed image and its measurable edge/line proxies.

**Gallery IDs and numeric probes** · full data: `dataset/_metadata/04_04_gallery_ids_and_numeric_probes.csv`

| dataset | sample_id | input_path | edge_fraction | hough_segments | long_segments | largest_contour_area_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| DocUNet | 3_2 | dataset/DocUNet/crop/3_2 copy.png | 0.11269 | 167 | 34 | 0.70892 |
| DocUNet | 5_2 | dataset/DocUNet/crop/5_2 copy.png | 0.08674 | 132 | 18 | 0.7935 |
| DocUNet | 13_2 | dataset/DocUNet/crop/13_2 copy.png | 0.11855 | 286 | 86 | 0.049 |
| DocUNet | 18_2 | dataset/DocUNet/crop/18_2 copy.png | 0.16261 | 448 | 126 | 0.00379 |
| DocUNet | 19_1 | dataset/DocUNet/crop/19_1 copy.png | 0.12933 | 318 | 139 | 0.08623 |
| DocUNet | 21_2 | dataset/DocUNet/crop/21_2 copy.png | 0.12705 | 360 | 118 | 0.13582 |
| DocUNet | 23_1 | dataset/DocUNet/crop/23_1 copy.png | 0.11293 | 419 | 124 | 0.00119 |
| DocUNet | 23_2 | dataset/DocUNet/crop/23_2 copy.png | 0.10345 | 405 | 137 | 0.00113 |
| DocUNet | 28_2 | dataset/DocUNet/crop/28_2 copy.png | 0.07425 | 118 | 27 | 0.00864 |
| DocUNet | 51_1 | dataset/DocUNet/crop/51_1 copy.png | 0.09069 | 157 | 103 | 0.76722 |
| DocUNet | 59_2 | dataset/DocUNet/crop/59_2 copy.png | 0.16955 | 488 | 122 | 0.80485 |
| DocUNet | 62_2 | dataset/DocUNet/crop/62_2 copy.png | 0.1241 | 387 | 147 | 0.12298 |

### 05.02 Raw grid inspection
Axes follow the original UVDoc loader; image-space values are **raw pixel coordinates**, not normalized neural-network targets. The 3D coordinate units are left as released because no physical unit calibration has been verified here.

**Grid structures and ranges** · full data: `dataset/_metadata/05_02_grid_structures_and_ranges.csv`

| dataset | sample_id | geom_name | raw_grid2d | loader_grid2d | raw_grid3d | loader_grid3d | x_min | x_max | y_min | y_max | z_min | z_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVDoc Final | 05470 | 01_00098_1_1_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 51.28518 | 443.04074 | 75.67259 | 598.68323 | -0.14254 | 0.14537 |
| UVDoc Final | 11816 | 12_00039_1_0_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 86.98148 | 448.46296 | 119.0395 | 649.88069 | -0.14742 | 0.14497 |
| UVDoc Final | 14074 | 04_00042_1_1_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 69.13332 | 438.2963 | 51.95431 | 587.76143 | -0.15007 | 0.14841 |
| UVDoc Benchmark | 00011 | 09_00014_1_0_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 15.88755 | 969.70813 | 15.49999 | 1257.93541 | -0.1552 | 0.15762 |
| UVDoc Benchmark | 00023 | 09_00028_1_0_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 5.5 | 937.49999 | 15.65647 | 1224.50005 | -0.14773 | 0.1431 |
| UVDoc Benchmark | 00039 | 11_00053_1_0_0 | (2, 61, 89) | (2, 89, 61) | (3, 61, 89) | (89, 61, 3) | 9.0 | 939.99999 | 12.99999 | 1188.99995 | -0.14303 | 0.1314 |

### 05.03 Image-space grid overlays
Corner labels A–D are array-index corners, not presumed document top-left/right labels. Use the saved figure to confirm visual correspondence. The companion table gives numerical coordinates and dimensions.

Figure: `runs/dataset_eda/figures/05_03.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/05_03_figure_data.csv`

| dataset | sample_id | geom_name | corner | x_px | y_px | image_width | image_height |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UVDoc Final | 05930 | 04_00022_1_1_0 | A | 22.22338 | 120.86121 | 488 | 712 |
| UVDoc Final | 05930 | 04_00022_1_1_0 | B | 470.15185 | 91.33854 | 488 | 712 |
| UVDoc Final | 05930 | 04_00022_1_1_0 | C | 77.27178 | 654.76182 | 488 | 712 |
| UVDoc Final | 05930 | 04_00022_1_1_0 | D | 460.21111 | 638.64704 | 488 | 712 |
| UVDoc Final | 06349 | 06_00053_1_0_1 | A | 63.71111 | 112.86504 | 488 | 712 |
| UVDoc Final | 06349 | 06_00053_1_0_1 | B | 422.29067 | 113.84769 | 488 | 712 |
| UVDoc Final | 06349 | 06_00053_1_0_1 | C | 61.23329 | 650.55838 | 488 | 712 |
| UVDoc Final | 06349 | 06_00053_1_0_1 | D | 424.17783 | 660.49747 | 488 | 712 |
| UVDoc Final | 14254 | 06_00013_1_1_0 | A | 26.38202 | 167.60913 | 488 | 712 |
| UVDoc Final | 14254 | 06_00013_1_1_0 | B | 459.53333 | 172.93117 | 488 | 712 |
| UVDoc Final | 14254 | 06_00013_1_1_0 | C | 50.57998 | 651.7587 | 488 | 712 |
| UVDoc Final | 14254 | 06_00013_1_1_0 | D | 416.15556 | 664.01503 | 488 | 712 |
| UVDoc Benchmark | 00010 | 09_00013_1_0_0 | A | 21.98304 | 19.18337 | 869 | 1180 |
| UVDoc Benchmark | 00010 | 09_00013_1_0_0 | B | 806.0057 | 34.10673 | 869 | 1180 |
| UVDoc Benchmark | 00010 | 09_00013_1_0_0 | C | 11.41291 | 1175.99994 | 869 | 1180 |
| UVDoc Benchmark | 00010 | 09_00013_1_0_0 | D | 860.00002 | 1133.64627 | 869 | 1180 |
| UVDoc Benchmark | 00048 | 11_00096_1_0_0 | A | 24.06599 | 68.0 | 875 | 1194 |
| UVDoc Benchmark | 00048 | 11_00096_1_0_0 | B | 819.92033 | 20.90807 | 875 | 1194 |
| UVDoc Benchmark | 00048 | 11_00096_1_0_0 | C | 63.289 | 1165.53941 | 875 | 1194 |
| UVDoc Benchmark | 00048 | 11_00096_1_0_0 | D | 826.34482 | 1126.45401 | 875 | 1194 |

### 06.02 Full unique-geometry audit
4,082 unique geometry variants are eligible for analysis. Plane residual is PCA deviation divided by 3D bbox diagonal. Neighbor normal change is a surface-shape proxy. Failures and outliers should be inspected on the image overlays rather than classified automatically as bad data.

**Geometry summary** · full data: `dataset/_metadata/06_02_geometry_summary.csv`

| dataset | grid_bbox_width_fraction_median | grid_bbox_width_fraction_mean | grid_bbox_width_fraction_min | grid_bbox_width_fraction_max | grid_bbox_height_fraction_median | grid_bbox_height_fraction_mean | grid_bbox_height_fraction_min | grid_bbox_height_fraction_max | outside_image_fraction_median | outside_image_fraction_mean | outside_image_fraction_min | outside_image_fraction_max | grid2d_step_median_px_median | grid2d_step_median_px_mean | grid2d_step_median_px_min | grid2d_step_median_px_max | grid2d_step_p95_px_median | grid2d_step_p95_px_mean | grid2d_step_p95_px_min | grid2d_step_p95_px_max | degenerate_cell_fraction_median | degenerate_cell_fraction_mean | degenerate_cell_fraction_min | degenerate_cell_fraction_max | orientation_inconsistent_fraction_median | orientation_inconsistent_fraction_mean | orientation_inconsistent_fraction_min | orientation_inconsistent_fraction_max | z_span_median | z_span_mean | z_span_min | z_span_max | plane_residual_normalized_median | plane_residual_normalized_mean | plane_residual_normalized_min | plane_residual_normalized_max | mean_neighbor_normal_angle_deg_median | mean_neighbor_normal_angle_deg_mean | mean_neighbor_normal_angle_deg_min | mean_neighbor_normal_angle_deg_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UVDoc Benchmark | 0.97269 | 0.97326 | 0.95592 | 0.98942 | 0.97942 | 0.97894 | 0.96574 | 0.99043 | 0.0 | 0.0 | 0.0 | 0.0 | 13.20886 | 13.31039 | 12.67779 | 14.37696 | 14.65527 | 14.84452 | 13.65888 | 16.97818 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.29292 | 0.29367 | 0.2693 | 0.32653 | 0.02282 | 0.02205 | 0.00437 | 0.04898 | 3.75392 | 3.72588 | 2.40337 | 5.12649 |
| UVDoc Final | 0.7881 | 0.79534 | 0.68746 | 0.96764 | 0.75635 | 0.76152 | 0.64738 | 0.93184 | 0.0 | 0.0 | 0.0 | 0.0 | 6.01064 | 6.03108 | 5.73524 | 6.51806 | 6.54583 | 6.62123 | 6.12743 | 7.92677 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00057 | 0.2945 | 0.29341 | 0.24849 | 0.34514 | 0.01556 | 0.01774 | 0.00199 | 0.06368 | 3.27187 | 3.3928 | 2.12941 | 7.04425 |

**Geometry failures** · full data: `dataset/_metadata/06_02_geometry_failures.csv`

_No rows available._

### 06.03 3D surface examples and extremes
3D wireframe visualizes the observed surface in arbitrary released coordinates. The table provides per-axis spans and sample IDs; the full extreme ranking is in `dataset/_metadata/uvdoc_geometry_extremes.csv`.

Figure: `runs/dataset_eda/figures/06_03.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/06_03_figure_data.csv`

| geom_name | sample_id | x_range | y_range | z_range |
| --- | --- | --- | --- | --- |
| 01_00107_1_0_1 | 01473 | 0.0365 | 0.20012 | 0.2974 |
| 10_00001_1_1_0 | 03378 | 0.0278 | 0.21408 | 0.29839 |
| 12_00023_1_1_0 | 03694 | 0.03616 | 0.21576 | 0.29504 |

### 07.02 Auxiliary data formats
Auxiliary array inspection sampled up to 32 unique geometries per release. Min/max values and array keys describe formats, not proven semantics.

**Auxiliary array inventory** · full data: `dataset/_metadata/07_02_auxiliary_array_inventory.csv`

| dataset | folder | key | shape | dtype | samples |
| --- | --- | --- | --- | --- | --- |
| UVDoc Benchmark | seg | seg | (1018, 1221) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (1021, 1286) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (1030, 1275) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (1043, 1217) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (785, 1347) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (812, 1161) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (813, 1184) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (843, 1309) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (861, 1222) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (862, 1161) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (862, 1173) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (867, 1166) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (869, 1180) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (870, 1307) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (875, 1194) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (881, 1206) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (881, 1309) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (890, 1185) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (899, 1262) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (917, 1190) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (925, 1091) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (927, 1217) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (928, 1303) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (931, 1237) | uint8 | 1 |
| UVDoc Benchmark | seg | seg | (947, 1300) | uint8 | 1 |

_Showing 25 of 66 rows; full CSV is saved separately._

**Release metadata keys** · full data: `dataset/_metadata/07_02_release_metadata_keys.csv`

| dataset | source | key | type | length |
| --- | --- | --- | --- | --- |
| UVDoc Final | split.json | textures | list | 3009 |
| UVDoc Final | split.json | backgrounds | list | 5640 |
| UVDoc Final | split.json | geoms | list | 1008 |
| UVDoc Final | 10_00010_1_1_1.json | sample_id | str | 10 |
| UVDoc Final | 10_00010_1_1_1.json | flip_horizontal | str | 4 |
| UVDoc Final | 10_00010_1_1_1.json | flip_vertical | str | 4 |
| UVDoc Final | 10_00010_1_1_1.json | sample_info | dict | 4 |
| UVDoc Final | 10_00010_1_1_1.json | cam | dict | 2 |
| UVDoc Benchmark | split.json | textures | list | 50 |
| UVDoc Benchmark | split.json | backgrounds | list | 50 |
| UVDoc Benchmark | split.json | geoms | list | 50 |
| UVDoc Benchmark | 09_00020_1_0_0.json | sample_id | str | 10 |
| UVDoc Benchmark | 09_00020_1_0_0.json | flip_horizontal | str | 5 |
| UVDoc Benchmark | 09_00020_1_0_0.json | flip_vertical | str | 5 |
| UVDoc Benchmark | 09_00020_1_0_0.json | sample_info | dict | 4 |
| UVDoc Benchmark | 09_00020_1_0_0.json | cam | dict | 2 |

**Auxiliary errors** · full data: `dataset/_metadata/07_02_auxiliary_errors.csv`

_No rows available._

### 07.03 UVDoc intermediate-image visual audit
The three image roles follow the filenames in the official dataset release. The table records the source of every image shown; no interpretation of color transfer is assumed.

Figure: `runs/dataset_eda/figures/07_03.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/07_03_figure_data.csv`

| sample_id | geom_name | role | path |
| --- | --- | --- | --- |
| 01765 | 03_00022_1_0_1 | Captured white geometry | img_geom/03_00022_1_0_1.png |
| 01765 | 03_00022_1_0_1 | Warped texture | warped_textures/01765.png |
| 01765 | 03_00022_1_0_1 | Rendered image | img/01765.png |
| 04824 | 00_00222_1_0_0 | Captured white geometry | img_geom/00_00222_1_0_0.png |
| 04824 | 00_00222_1_0_0 | Warped texture | warped_textures/04824.png |
| 04824 | 00_00222_1_0_0 | Rendered image | img/04824.png |
| 06349 | 06_00053_1_0_1 | Captured white geometry | img_geom/06_00053_1_0_1.png |
| 06349 | 06_00053_1_0_1 | Warped texture | warped_textures/06349.png |
| 06349 | 06_00053_1_0_1 | Rendered image | img/06349.png |
| 15005 | 07_00102_1_0_1 | Captured white geometry | img_geom/07_00102_1_0_1.png |
| 15005 | 07_00102_1_0_1 | Warped texture | warped_textures/15005.png |
| 15005 | 07_00102_1_0_1 | Rendered image | img/15005.png |

### 08.02 Photometric domain comparison
Figure highlights P5/median/P95 contrast proxy; companion table includes the other descriptors. Different sample sizes are visible in `count`, and no statistical significance or causal effect is asserted.

Figure: `runs/dataset_eda/figures/08_02.png`. The table below is the numeric/text companion for this figure.

**Figure data** · full data: `dataset/_metadata/08_02_figure_data.csv`

| dataset | descriptor | count | p05 | median | p95 |
| --- | --- | --- | --- | --- | --- |
| DIR300 | luminance_mean | 300 | 102.55728 | 132.73492 | 160.68003 |
| DIR300 | luminance_std | 300 | 34.68175 | 56.21854 | 77.9849 |
| DIR300 | saturation_mean | 300 | 17.55944 | 55.23829 | 100.37539 |
| DIR300 | edge_fraction | 300 | 0.05189 | 0.09908 | 0.18646 |
| DIR300 | tile_luminance_range | 300 | 48.2248 | 96.21686 | 144.18021 |
| DIR300 | laplacian_variance | 300 | 277.85862 | 580.00436 | 1193.63567 |
| DocUNet | luminance_mean | 130 | 123.23252 | 164.96359 | 193.25395 |
| DocUNet | luminance_std | 130 | 31.08499 | 54.26761 | 75.97007 |
| DocUNet | saturation_mean | 130 | 26.94088 | 50.42114 | 83.25912 |
| DocUNet | edge_fraction | 130 | 0.06557 | 0.12913 | 0.20937 |
| DocUNet | tile_luminance_range | 130 | 35.09541 | 79.2834 | 132.38061 |
| DocUNet | laplacian_variance | 130 | 483.28694 | 998.55606 | 1859.02451 |
| UVDoc Benchmark | luminance_mean | 50 | 118.01391 | 146.81371 | 166.10469 |
| UVDoc Benchmark | luminance_std | 50 | 25.66117 | 39.59435 | 51.69307 |
| UVDoc Benchmark | saturation_mean | 50 | 26.98584 | 66.20444 | 118.98475 |
| UVDoc Benchmark | edge_fraction | 50 | 0.04276 | 0.11122 | 0.15591 |
| UVDoc Benchmark | tile_luminance_range | 50 | 38.88943 | 70.81786 | 114.99269 |
| UVDoc Benchmark | laplacian_variance | 50 | 186.64226 | 522.91778 | 1503.35914 |
| UVDoc Final | luminance_mean | 2400 | 41.05587 | 125.67575 | 160.15945 |
| UVDoc Final | luminance_std | 2400 | 25.59498 | 43.57277 | 64.44388 |
| UVDoc Final | saturation_mean | 2400 | 12.67507 | 56.27506 | 171.54727 |
| UVDoc Final | edge_fraction | 2400 | 0.04395 | 0.12901 | 0.23343 |
| UVDoc Final | tile_luminance_range | 2400 | 28.58524 | 65.49452 | 117.60105 |
| UVDoc Final | laplacian_variance | 2400 | 123.24256 | 486.30631 | 1588.7306 |

### 08.03 Annotated geometry comparison
Geometric descriptors are from supplied UVDoc annotations only. Their distributions measure differences between training and benchmark geometry, not outcomes of an unwarping model.

**Geometry distributions by release** · full data: `dataset/_metadata/08_03_geometry_distributions_by_release.csv`

| dataset | descriptor | n | p05 | median | p95 |
| --- | --- | --- | --- | --- | --- |
| UVDoc Benchmark | plane_residual_normalized | 50 | 0.00618 | 0.02282 | 0.04538 |
| UVDoc Benchmark | z_span | 50 | 0.27453 | 0.29292 | 0.31315 |
| UVDoc Benchmark | mean_neighbor_normal_angle_deg | 50 | 2.64133 | 3.75391 | 4.99659 |
| UVDoc Benchmark | orientation_inconsistent_fraction | 50 | 0.0 | 0.0 | 0.0 |
| UVDoc Final | plane_residual_normalized | 4032 | 0.0045 | 0.01556 | 0.03793 |
| UVDoc Final | z_span | 4032 | 0.28004 | 0.2945 | 0.3035 |
| UVDoc Final | mean_neighbor_normal_angle_deg | 4032 | 2.39719 | 3.27186 | 4.86195 |
| UVDoc Final | orientation_inconsistent_fraction | 4032 | 0.0 | 0.0 | 0.0 |

### 09.02 Real benchmark reference compatibility
Aspect/log-ratio and portrait/landscape differences are descriptive warnings; they are not automatic errors or reasons to exclude examples. DocUNet 64_1 and 64_2 are explicitly flagged for manual inspection.

**Benchmark pair summary** · full data: `dataset/_metadata/09_02_benchmark_pair_summary.csv`

| dataset | n | orientation_category_mismatches | flagged_for_visual_review | median_abs_log_aspect_ratio |
| --- | --- | --- | --- | --- |
| DIR300 | 300 | 0 | 0 | 0.06995 |
| DocUNet | 130 | 2 | 2 | 0.07484 |

**Potential orientation/category cases** · full data: `dataset/_metadata/09_02_potential_orientation_category_cases.csv`

| dataset | sample_id | input_path | reference_path | aspect_log_ratio | needs_visual_review |
| --- | --- | --- | --- | --- | --- |
| DocUNet | 11_1 | dataset/DocUNet/crop/11_1 copy.png | dataset/DocUNet/scan/11.png | -0.03596 | False |
| DocUNet | 11_2 | dataset/DocUNet/crop/11_2 copy.png | dataset/DocUNet/scan/11.png | -0.11614 | False |
| DocUNet | 64_1 | dataset/DocUNet/crop/64_1 copy.png | dataset/DocUNet/scan/64.png | 0.03735 | True |
| DocUNet | 64_2 | dataset/DocUNet/crop/64_2 copy.png | dataset/DocUNet/scan/64.png | 0.16174 | True |

### 09.03 Real benchmark paired gallery
Gallery: `runs/dataset_eda/galleries/benchmark_reference_pairs.jpg`. The companion table records all displayed pairs and dimensions. Inspect orientation and content manually before setting an evaluation alignment rule.

**Displayed pairs** · full data: `dataset/_metadata/09_03_displayed_pairs.csv`

| dataset | sample_id | input_path | reference_path | input_width | input_height | reference_width | reference_height | aspect_log_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DocUNet | 64_1 | dataset/DocUNet/crop/64_1 copy.png | dataset/DocUNet/scan/64.png | 1354 | 1688 | 2550 | 3300 | 0.03735 |
| DocUNet | 64_2 | dataset/DocUNet/crop/64_2 copy.png | dataset/DocUNet/scan/64.png | 1517 | 1670 | 2550 | 3300 | 0.16174 |
| DocUNet | 2_2 | dataset/DocUNet/crop/2_2 copy.png | dataset/DocUNet/scan/2.png | 1365 | 2421 | 944 | 2702 | 0.4786 |
| DocUNet | 4_2 | dataset/DocUNet/crop/4_2 copy.png | dataset/DocUNet/scan/4.png | 796 | 1609 | 677 | 1986 | 0.37244 |
| DocUNet | 10_1 | dataset/DocUNet/crop/10_1 copy.png | dataset/DocUNet/scan/10.png | 1724 | 1241 | 2506 | 1262 | -0.35726 |
| DocUNet | 2_1 | dataset/DocUNet/crop/2_1 copy.png | dataset/DocUNet/scan/2.png | 1155 | 2352 | 944 | 2702 | 0.34046 |
| DIR300 | 8 | dataset/DIR300/_source/DIR300/dist/8.png | dataset/DIR300/_source/DIR300/gt/8.png | 3013 | 3795 | 1620 | 2296 | 0.11799 |
| DIR300 | 130 | dataset/DIR300/_source/DIR300/dist/130.png | dataset/DIR300/_source/DIR300/gt/130.png | 3072 | 4096 | 1788 | 2524 | 0.05707 |
| DIR300 | 191 | dataset/DIR300/_source/DIR300/dist/191.png | dataset/DIR300/_source/DIR300/gt/191.png | 3072 | 4096 | 1704 | 2380 | 0.04644 |
| DIR300 | 221 | dataset/DIR300/_source/DIR300/dist/221.png | dataset/DIR300/_source/DIR300/gt/221.png | 3459 | 4312 | 1636 | 2328 | 0.13233 |

### 10.02 Proposed grouped split
Validation groups: 101 of 1,008, seed=20261120. The last two geometry ID bits enumerate four variants per base; all variants are kept together. Repeated textures/backgrounds are quantified rather than incorrectly claimed absent.

**Split sizes** · full data: `dataset/_metadata/10_02_split_sizes.csv`

| split | images | geometry_variants | base_captures | textures | backgrounds |
| --- | --- | --- | --- | --- | --- |
| train | 17992 | 3628 | 907 | 3001 | 5410 |
| val | 2008 | 404 | 101 | 1475 | 1681 |

**Split source overlap** · full data: `dataset/_metadata/10_02_split_source_overlap.csv`

| identity | shared |
| --- | --- |
| base_geometry | 0 |
| geom_name | 0 |
| texture_name | 1473 |
| background_name | 1621 |

### 10.03 Train/validation geometry balance
The reported distributions describe the fixed capture-grouped split. If severe imbalance appears, document it rather than repeatedly searching for a seed that improves test scores.

**Geometry descriptors by split** · full data: `dataset/_metadata/10_03_geometry_descriptors_by_split.csv`

| descriptor | split | n_unique_geometries | median | p05 | p95 |
| --- | --- | --- | --- | --- | --- |
| plane_residual_normalized | train | 3628 | 0.01577 | 0.00491 | 0.03793 |
| plane_residual_normalized | val | 404 | 0.0138 | 0.00433 | 0.03666 |
| z_span | train | 3628 | 0.29449 | 0.28027 | 0.30323 |
| z_span | val | 404 | 0.29474 | 0.27967 | 0.30585 |
| mean_neighbor_normal_angle_deg | train | 3628 | 3.27637 | 2.4016 | 4.86202 |
| mean_neighbor_normal_angle_deg | val | 404 | 3.23805 | 2.39448 | 4.60092 |
| orientation_inconsistent_fraction | train | 3628 | 0.0 | 0.0 | 0.0 |
| orientation_inconsistent_fraction | val | 404 | 0.0 | 0.0 | 0.0 |

### 10.04 Exported train and validation definitions
No original images/grids were moved or modified. This is a proposed reproducible local validation split; it is not the training/validation protocol used in the original UVDoc paper.

**Split output files** · full data: `dataset/_metadata/10_04_split_output_files.csv`

| file | rows |
| --- | --- |
| dataset/_metadata/uvdoc_train.csv | 17992 |
| dataset/_metadata/uvdoc_val.csv | 2008 |
| dataset/_metadata/split_spec.json | 1 |

### 11.02 Research-readiness conclusions and limitations
EDA is now summarized independently of notebook plot outputs. Figures and numerical companions, sampling limits, split policy, and outstanding issues are recorded above. Visual manual-review labels remain unfilled until somebody actually inspects those examples.

**Method implications** · full data: `dataset/_metadata/11_02_method_implications.csv`

| research_area | finding_or_next_check |
| --- | --- |
| Deep-learning supervision | Raw 2D/3D grid load and normalization conventions are distinguished; visually verify grid overlays. |
| Validation design | Use disjoint base captures, not independent rendered images or only full geometry variants. |
| Held-out evaluation | UVDoc Benchmark, DocUNet and DIR300 are excluded from the train/validation split. |
| Classical boundaries | Inspect Canny/contour galleries and fill the manual-review worksheet; automatic contours are not labels. |
| Classical text lines | Use line-candidate statistics as probes; manually verify whether lines correspond to printed text. |
| Reference compatibility | Inspect flagged orientation/content pairs, including DocUNet 64_1 and 64_2, before formal metric evaluation. |
| Synthetic to real | Use recorded domain differences as hypotheses, not proven causes of model errors. |
| Dataset completeness | Re-run actual path and annotation checks whenever temporary Colab storage resets. |

**Unresolved issues** · full data: `dataset/_metadata/11_02_unresolved_issues.csv`

| outstanding_issue |
| --- |
| No automatic audit failures observed; human visual checks still required. |

# Geometry conventions (Stage 2)

**Status:** Implemented with CPU synthetic tests. No benchmark score is implied.

## Coordinates and mapping direction

- Image arrays have dimensions `(height, width[, channels])`.
- Coordinates are always `(x, y)`, where `x` points right, `y` points down, and `(0, 0)` is the center of the upper-left pixel.
- The dense backward map is `(H_output, W_output, 2)` and contains the **source** position for each **output** position.
- `map_space="source_pixels"` means values are measured in source pixels. `map_space="normalized_minus1_plus1"` means PyTorch sampling coordinates.
- The canonical UVDoc interpolation uses PyTorch `align_corners=True`.
- `models/interfaces.py` is the authoritative Stage 1 result contract. Its `backward_map` is dense only; predicted coarse grids are separate artifacts.

For source image width `W`, height `H`, and `align_corners=True`:

```
x_normalized = 2 * x_pixels / (W - 1) - 1
x_pixels     = (x_normalized + 1) * (W - 1) / 2

y_normalized = 2 * y_pixels / (H - 1) - 1
y_pixels     = (y_normalized + 1) * (H - 1) / 2
```

With `align_corners=False`, the pixel-center offset of `0.5` changes these formulas. The implementation supports both but rejects single-pixel dimensions for `align_corners=True`.

## Forward matrices and backward sampling maps

A homography `H` estimated by `estimate_homography(source, target)` maps *source points to output points*. The backward map used for image sampling is constructed from `H^{-1}`, not `H`. `homography_backward_map(H, source_size=(W,H), output_size=(W,H))` returns an explicit pixel-space map without modifying an image.

`sample_backward(image, backward_map, map_space)` accepts a 2D or HWC NumPy image, uses PyTorch `grid_sample`, and returns floating HWC values at the original image intensity scale. It does **not** round, clip, resize the input, or silently choose another coordinate space. Supported baseline sampling modes are bilinear and nearest; UVDoc itself is strictly bilinear/zero-padding.

## Composition of inverse maps

If `B_A : output1 -> original` and `B_B : output2 -> output1`, then the composition `B_A(B_B(p)) : output2 -> original` is returned by `compose_backward_maps(B_A, B_B, source_size)`. The function also returns a boolean validity mask, propagating invalid source-sampling positions. Invalid locations have placeholder coordinates `(0,0)` and must not be treated as geometrically valid. The interpolation of coordinates uses OpenCV linear resampling, which has finite interpolation precision; tests allow a corresponding numerical tolerance.

`sampling_valid_mask` refers only to coordinates within source image bounds. It does not detect document content or judge rectification quality. For `align_corners=False`, the valid *pixel-center* normalized interval excludes `-1` and `1`, which would otherwise sample padding.

## UVDoc grid distinctions

| Object | Typical shape | Interpretation |
|---|---|---|
| UVDoc predicted 2D grid | `(1,2,45,31)` | Normalized coarse backward coordinates |
| UVDoc predicted 3D grid | `(1,3,45,31)` | Auxiliary three-channel prediction; not used for warping |
| Upsampled UVDoc grid | `(1,2,H_source,W_source)` | Normalized per-output sampling coordinates |
| Dense backward map | `(H_source,W_source,2)` | Same spatial sampling coordinates in HWC form |
| UVDoc dataset 2D annotation | `(2,89,61)` in Stage 1 reader | Released target annotation at another resolution |
| UVDoc dataset 3D annotation | `(89,61,3)` in Stage 1 reader | Released 3D annotation, different frame/resolution |

Predictions and released annotation grids must not be directly subtracted without a separate validated protocol. Implementing that protocol belongs to Stage 3, not Stage 2.

## Color and pixel conversion

The shared geometry functions operate on whatever numeric color scale the caller supplies. The official UVDoc reference adapter uses OpenCV `imread(..., IMREAD_COLOR)` followed by BGR to RGB, converts to float32 `[0,1]`, resizes the **network input only** to `(488,712)` using `cv2.INTER_LINEAR`, then warps the original-resolution float RGB image with `F.interpolate` and `F.grid_sample` using `align_corners=True`, bilinear sampling, and zero padding. Final uint8 output uses `(... * 255).astype(np.uint8)` (truncation). Alternative interpolation, EXIF policy, CLAHE, and output resizing are not enabled as hidden defaults.

## Tests and limitations

Geometry tests are CPU-executable, use deterministic synthetic examples, and cover corner conversion, coordinate identity, known displacement, homographies, invalid grids, boundary padding, map composition, and mapping validity. Image-output parity against the official pretrained checkpoint is a separate Colab gate requiring the pinned upstream source and weights. Neither a synthetic pass nor a plausible-looking output establishes official benchmark equivalence.

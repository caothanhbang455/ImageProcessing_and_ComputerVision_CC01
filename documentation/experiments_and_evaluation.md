# Designing and Evaluating Document Unwarping Experiments

A corrected document can look better without being geometrically more accurate. An image may have straighter text but lose part of the page. Another may preserve the full page while retaining some curvature. Sharpening or contrast adjustment can improve its appearance without changing the geometric correction at all.

This is why comparing document-unwarping methods requires more than placing the outputs side by side or reporting one average score. A useful experiment asks a specific question, uses suitable data, measures the relevant properties, and records what happened to every attempted image. The conclusions should follow from that evidence, including cases where a method fails or a metric cannot be computed.

This document explains how to plan that comparison and interpret the results. The [Implementation guide](implementation_guide.md) covers how to run methods and save predictions; [Evaluation protocols](reference/evaluation_protocols.md) defines the exact benchmark procedures.

## 1. Start with a question that can be tested

Before selecting a metric, decide what the experiment is trying to establish.

Suppose a classical perspective correction and a pretrained neural model both produce a flatter-looking page. The relevant question might be which one better straightens text, which one preserves the full page, or which one succeeds more often when the borders are difficult to find. These are different questions. They may have different answers even for the same pair of methods.

Consider another example: applying contrast enhancement before UVDoc. If the final image looks clearer, that alone does not show that the predicted geometry has improved. Contrast enhancement directly changes pixel appearance. To test a geometric claim, the experiment needs geometric evidence as well as any image-similarity measurement.

A good question identifies the comparison, the intended improvement, and the conditions under which it will be measured. Examples include:

- Does automatic homography straighten nearly planar pages as reliably as a pretrained neural method on the same inputs?
- Does changing one preprocessing operation improve the fixed neural baseline's geometry, rather than merely its appearance?
- Does a candidate method preserve more page content without increasing local stretching?

Not every experiment needs several metrics. It does need measurements suited to its question. A runtime experiment, for example, must define what is timed and on which hardware. It cannot answer a geometric-accuracy question by itself.

## 2. Choose data that can answer the question

The available datasets serve different purposes. They should not be treated as interchangeable collections of document pictures.

| Dataset | Useful for | Important limitation |
|---|---|---|
| **UVDoc Final** | Developing and diagnosing methods that use 2D/3D geometric annotations | The project's train/validation split separates base geometry groups, but some texture or background content overlaps |
| **UVDoc Benchmark** | Evaluating geometric line straightness and prepared texture results | Only 50 examples; its line metric needs a compatible predicted geometric grid |
| **DocUNet** | Real-world rectification using photographed pages and reference scans | 130 photographs represent 65 documents; document 64 has a known orientation issue |
| **DIR300** | Evaluation on a different set of real photographed documents | 300 numbered input/reference pairs; benchmark preparation differs from DocUNet |

A **training set** is used to fit model parameters. A **validation set** helps choose a configuration or decide when to stop training. **Exploratory examples** are images inspected while developing or debugging a method. A **benchmark set** is used to assess the finished method under a declared procedure.

These uses can overlap in practice, but the overlap must be disclosed. Repeatedly examining particular benchmark images and tuning a method to them changes what the resulting scores demonstrate. Those examples can still be included in an officially defined full-benchmark run, but they should not be described as untouched held-out cases.

The UVDoc Final split is a useful illustration. Its 20,000 images are grouped by underlying geometry, with 17,992 training and 2,008 validation examples. The split avoids sharing a base geometry between training and validation, but it does not guarantee independent texture and background content. It also does not establish that the released official UVDoc checkpoint never encountered these images during its own training.

For DocUNet, the two views of each underlying document are related observations. Treating all 130 images as completely independent pages can exaggerate how much distinct document content has been tested. Use sample-level results for the benchmark, but consider document-level grouping when discussing consistency or uncertainty.

Pair images by their stable dataset identities, not by position in a directory listing. The exact file pairing and split rules are documented in [Dataset protocols](reference/dataset_protocols.md).

## 3. Keep comparisons fair without forcing methods to be identical

A fair comparison starts with the **same selected input samples** and a clearly identified evaluation protocol. It does not require every method to have the same internal architecture, input preprocessing, or output dimensions.

A homography-based method may return a tightly cropped rectangular page. A learned model may retain the dimensions of the original photograph. These images cannot be compared fairly by choosing a different ad hoc resize or crop for whichever output looks worse. The benchmark's alignment and sizing rules must be applied consistently after inference.

Reference images also need careful handling. A flat reference scan can be used by a permitted evaluation procedure to align and score a prediction. It should not be used secretly during inference to find better page corners or to select a favorable crop, unless the experiment is explicitly defined as a template-guided method.

Record the factors that could change a result: input variant, selected samples, preprocessing, checkpoint or model version, output size, interpolation, evaluation protocol, and alignment procedure. For runtime comparisons, record the device and timing boundaries as well.

When investigating an improvement, change one factor at a time where practical. For example, to compare interpolation methods, hold the predicted backward map fixed and change only the image sampler. If the map and sampler both change, an apparent improvement cannot be attributed to either one.

Likewise, a perspective-correction-plus-UVDoc experiment should be compared with the unchanged official UVDoc model on the same images. The additional correction changes the model input and may introduce another resampling step. Better-looking outputs alone do not establish that the extra stage is beneficial.

## 4. Understand what the measurements mean

No single metric describes every useful property of a corrected document.

| Measurement | Main question | Preferred direction |
|---|---|---|
| **MS-SSIM** | How similar is the image structure to a reference under the selected evaluation procedure? | Higher |
| **Local Distortion (LD)** | How much local mismatch remains under the benchmark's correspondence and distortion procedure? | Lower |
| **Aligned Distortion (AD)** | How much distortion remains under the selected aligned-reference procedure? | Lower |
| **UVDoc line deviation** | How far do known straight horizontal and vertical lines depart from straightness after correction? | Lower |
| **OCR Character Error Rate (CER)** | How many recognition edits are needed relative to a suitable reference transcription? | Lower |
| **Runtime** | How long does a specified part of the method take under stated conditions? | Lower |

**Image similarity is not the same as geometric accuracy.** Two images with similar geometry may receive different similarity scores because one has sharper text, different brightness, or a slightly different crop. A method that improves MS-SSIM therefore needs additional evidence before claiming an improvement in geometric correction.

**Straight lines are not the entire page.** A correction might straighten visible text but stretch letters near the margins or cut off the bottom of the document. Line deviation captures a specific geometric property, not completeness or readability.

**LD and AD are defined by their evaluation procedures.** They should not be described as ordinary pixelwise differences. The historically used reference procedures include alignment and, for relevant LD calculations, SIFT-flow-based correspondence. Replacing them with a different optical-flow algorithm or a convenient Python function creates a different metric, even if the result has a similar name.

OCR can add useful evidence about readability, but it requires a valid text reference, a declared recognition system, and consistent normalization rules. OCR-derived text from a scan is not automatically verified ground truth. In this project, OCR is a possible additional analysis, not a claimed completed reference benchmark.

Efficiency is similarly conditional. A GPU model timed only for its forward pass is not directly comparable with a CPU method timed from image loading through output saving. Decide whether the experiment measures model execution, geometric sampling, preprocessing, or end-to-end latency, and use the same boundary for each comparison.

## 5. Use an evaluation procedure that matches the outputs

The datasets do not all use the same reference evaluation. The project identifies procedures by protocol ID in `configs/evaluation.yaml`; the technical definitions are in [Evaluation protocols](reference/evaluation_protocols.md).

| Dataset | Available evaluation path | Requirement or limitation |
|---|---|---|
| UVDoc Benchmark | `uvdoc_lines_official_n25_v1` | Requires the predicted coarse 2D backward grid in the expected UVDoc representation |
| UVDoc Benchmark | `uvdoc_texture_prepare_v1` and `uvdoc_visual_matlab_v1` | Texture preparation can run in Python; reference visual scores need MATLAB and its dependencies |
| DocUNet | `docunet_matlab_legacy_v1` | Uses cropped inputs and the selected MATLAB MS-SSIM/LD/AD procedure |
| DocUNet | `docunet_matlab_corrected64_v1` | Requires separately generated predictions for correctly oriented document-64 inputs |
| DIR300 | `dir300_matlab_v1` | Uses the selected MATLAB MS-SSIM/LD procedure; AD is not part of this particular wrapper |

This compatibility check matters whenever methods produce different kinds of output. An image-only method can participate in supported image-reference preparation, but it cannot automatically receive the UVDoc line metric. That test needs the model's coarse geometric grid. A dense backward map returned by an independent method is not interchangeable with UVDoc's coarse grid without a separately verified conversion and evaluation procedure.

The DocUNet document-64 issue also needs an explicit choice. Samples `64_1` and `64_2` have a known 180-degree orientation mismatch relative to their reference scans. The corrected protocol requires inference from the appropriately oriented input; rotating an existing prediction afterward is not assumed to reproduce the same model behavior. The legacy and corrected procedures should therefore not be mixed in one unnamed results column.

The selected DIR300 reference wrapper calculates MS-SSIM and LD. Although other papers report additional metrics on DIR300, those numbers may come from different evaluation code. They cannot be silently attributed to this wrapper.

One more distinction is essential: **preparing data for reference evaluation is not the same as computing a reference score**. The Python code can arrange images, filenames, and reference inputs even when MATLAB is unavailable. In that situation, the missing benchmark score must remain unavailable. Substituting a different Python similarity function and labeling it as the published result would be misleading.

The recorded Colab verification completed the UVDoc line evaluation for all 50 benchmark examples. Independent comparison with the pinned official line evaluator passed on samples `00010` and `00011`, with zero reported difference for those checked lines. The corresponding texture preparation completed for 50 examples, with two-sample pixel parity checks. Published MATLAB visual scores were not computed in that environment because the MATLAB Engine was unavailable. These observations verify specific implementation steps, not a claim of superior unwarping quality.

## 6. Compare both the scores and the number of usable results

A method can have a good average score simply because its difficult cases were excluded. To avoid this, keep four counts separate:

1. **Attempted:** inputs submitted to the method.
2. **Successful:** inputs for which the method produced an accepted prediction.
3. **Eligible:** successful outputs that meet the selected metric's requirements and have the necessary reference data.
4. **Valid scores:** eligible outputs for which the metric actually returned a finite value.

If the reference dependency is missing, the number of valid published scores may be zero even when all predictions were generated and prepared correctly. A missing score should not be entered as zero, because zero may have a real numerical meaning for the metric.

Here is a hypothetical example, not a result from this project:

| Method | Attempted | Successful | Valid scores | Mean score on valid cases |
|---|---:|---:|---:|---:|
| A | 20 | 20 | 20 | 0.70 |
| B | 20 | 16 | 16 | 0.74 |

If the scores are MS-SSIM, the table does **not** establish that B is generally better. B has no scores for four inputs, and its mean was calculated on a different set of cases. A fair paired comparison would evaluate A and B on the same 16 eligible image identities, then report the four missing B results separately. It should also inspect why those cases failed.

For each metric, a mean and a median provide useful but different summaries. The mean is sensitive to unusually poor or strong scores; the median describes the middle of the distribution. The spread, worst cases, and counts of improved or worsened images can reveal changes that the average hides.

For paired methods, calculate differences for matched sample identities rather than comparing two unrelated means. The direction of improvement depends on the metric: higher MS-SSIM is better, while lower LD or line deviation is better. Small differences on a few selected examples should not be presented as evidence of general superiority.

More detailed uncertainty analysis may be appropriate for a final comparison, especially when observations share the same document. It is not necessary to turn every small diagnostic experiment into a formal statistical study. The first requirement is to show what was actually measured and on which images.

## 7. Inspect what the images reveal

Numerical scores are useful for summarizing many results, but they rarely explain why a correction failed. Visual inspection should include the original image, the prediction, and the flat reference when one exists. A displayed alignment or page outline can also help identify the source of a problem.

| Observation | Possible issue to investigate |
|---|---|
| Wrong page outline or background included | Page detection selected the wrong region |
| Page still trapezoidal | Residual perspective distortion |
| Text baselines remain bent | Incomplete nonrigid correction |
| Characters look stretched or compressed | Local mapping distortion |
| Part of the page disappears | Cropping or invalid output coverage |
| Repeated, folded, or collapsed regions | Invalid geometric mapping or sampling |
| Text becomes soft or jagged | Interpolation, repeated resampling, or resolution loss |
| Prediction disagrees strongly with the scan despite looking plausible | Possible reference alignment or orientation issue |

These are starting points for inspection, not labels that can be assigned automatically from a low metric value. A poor MS-SSIM score, for example, cannot distinguish residual curvature from different lighting or an incorrectly aligned reference.

Separating the failure source is especially important for classical methods. Failure to find four corners is a page-detection failure. Finding the correct corners but leaving curved text is a limitation of the chosen geometric model. Both may lead to a poor output, but they call for different changes.

Choose representative successful cases as well as difficult ones. A comparison that displays only the most attractive outputs is not enough. Include cases where the methods disagree, where both fail, and where one metric improves while another becomes worse.

## 8. Report only what the experiment establishes

A useful result summary does not need elaborate formatting. It should make the experiment understandable without requiring the original notebook: state the question, identify the compared methods and configurations, name the dataset and input variant, give the evaluation protocol, report coverage and scores, and describe the main visual observations.

For each conclusion, distinguish three kinds of evidence:

**Implementation verification** establishes that a function behaves as intended on tests or matches a reference implementation on selected cases. Exact parity is strong evidence for that particular operation, but it does not establish the quality of the model's predictions.

**Experimental measurement** establishes what happened on a stated set of images with a stated configuration and metric. It can support a comparison within those conditions.

**Broader research claims** require stronger evidence across suitable data and conditions. A method that improves a few known examples has not necessarily improved generalization to unseen documents.

For example, an appropriate conclusion from a hypothetical small experiment might be:

> On the selected nearly planar pages, automatic homography produced acceptable correction when the page boundary was detected accurately. Several failures came from incorrect boundaries, while the neural method retained more usable results on those cases. This comparison does not establish a general advantage on curved pages or on the full benchmark.

This conclusion says what was observed, identifies the likely reason, and limits the claim to the tested conditions. A complete report would accompany it with sample counts, metric results when available, and examples supporting the observations.

Saved inference and evaluation runs provide the evidence for such a report. Keep the method configuration, sample identities, scores, unavailable measurements, and failure reasons together so later comparisons can be checked without rerunning the models.

For the exact calculation and reference compatibility rules, consult [Evaluation protocols](reference/evaluation_protocols.md). For practical execution and saved-file locations, use the [Implementation guide](implementation_guide.md). The purpose of experimental evaluation is not to produce the largest number of scores. It is to determine what a method changes, how reliably it does so, and whether the evidence supports the proposed improvement.
